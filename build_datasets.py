"""
Build the Objective 2 datasets (FIFA World Cup 2026) from matches.csv.

Outputs
  lr21_match_dataset.csv     104 rows (one per match)       target: goal_diff
  lr22_teammatch_dataset.csv 208 rows (one per team-match)  target: goals_scored

Every explanatory variable is known BEFORE kick-off:
  - FIFA ranking published 11 June 2026 (pre-tournament)
  - venue / hosting country, stage of tournament, rest days
  - averages of statistics from matches kicked off EARLIER in the tournament

Rolling averages use empirical-Bayes shrinkage so teams with 0-2 earlier
matches are pulled toward the tournament-to-date league mean:
    shrunk_avg = (sum_prior + K * prior_mean) / (n_prior + K),   K = 2
For the opening matches (no earlier data at all) a fixed fallback constant is used.
"""
import re
import numpy as np
import pandas as pd

K = 2
REST_CAP = 7            # days; first match of a team is treated as fully rested

# FIFA/Coca-Cola Men's World Ranking, 11 June 2026 (ranks as published by FIFA / ESPN / MLSsoccer).
RANK = {
    "Argentina": 1, "Spain": 2, "France": 3, "England": 4, "Portugal": 5, "Brazil": 6,
    "Morocco": 7, "Netherlands": 8, "Belgium": 9, "Germany": 10, "Croatia": 11,
    "Colombia": 13, "Mexico": 14, "Senegal": 15, "Uruguay": 16, "United States": 17,
    "Japan": 18, "Switzerland": 19, "IR Iran": 20, "Türkiye": 22, "Ecuador": 23,
    "Austria": 24, "Korea Republic": 25, "Australia": 27, "Algeria": 28, "Egypt": 29,
    "Canada": 30, "Norway": 31, "Côte d'Ivoire": 33, "Panama": 34, "Sweden": 38,
    "Czechia": 40, "Paraguay": 41, "Scotland": 42, "Tunisia": 45, "Congo DR": 46,
    "Uzbekistan": 50, "Qatar": 56, "Iraq": 57, "South Africa": 60, "Saudi Arabia": 61,
    "Jordan": 63, "Bosnia–Herz": 64, "Cabo Verde": 67, "Ghana": 73, "Curaçao": 82,
    "Haiti": 83, "New Zealand": 85,
}
assert len(RANK) == 48

HOST_COUNTRY = {"Mexico": "MEX", "Canada": "CAN", "United States": "USA"}
MEX_VENUES = ("Estadio Banorte", "Estadio Akron", "Estadio BBVA")
CAN_VENUES = ("BC Place", "BMO Field")

# fallbacks for the very first matches (no earlier data): typical World Cup values
FALLBACK = dict(gf=1.3, ga=1.3, sot=4.0, poss=50.0, shots=12.0, corners=4.5, intercept=8.0)

# ------------------------------------------------------------------ load & clean
m = pd.read_csv("matches.csv")
m["kickoff"] = pd.to_datetime(m["date"] + " " + m["start_time"])
m = m.sort_values(["kickoff"], kind="stable").reset_index(drop=True)
m["match_id"] = np.arange(1, len(m) + 1)


def parse_score(s):
    """'2–1' -> (2,1); '(3)1–1(4)' -> (1,1)  (penalty shoot-out excluded)."""
    nums = re.sub(r"\(\d+\)", "", s)
    a, b = re.split(r"[–-]", nums)
    return int(a), int(b)


m[["hg", "ag"]] = m["score"].apply(lambda s: pd.Series(parse_score(s)))


def venue_country(v):
    if v.startswith(MEX_VENUES):
        return "MEX"
    if v.startswith(CAN_VENUES):
        return "CAN"
    return "USA"


m["venue_country"] = m["venue"].apply(venue_country)
m["knockout"] = (m["round"] != "Group stage").astype(int)

# ------------------------------------------------------------------ long format (team-match)
stats = {  # name -> (home col, away col)
    "sot": ("home_sot", "away_sot"),
    "poss": ("home_possession", "away_possession"),
    "shots": ("home_total_shots", "away_total_shots"),
    "corners": ("home_corners", "away_corners"),
    "intercept": ("home_interceptions", "away_interceptions"),
}
rows = []
for _, r in m.iterrows():
    for side, opp in (("home", "away"), ("away", "home")):
        d = dict(match_id=r.match_id, kickoff=r.kickoff, date=pd.to_datetime(r.date),
                 team=r[f"{side}_team"], opp=r[f"{opp}_team"], side=side,
                 knockout=r.knockout,
                 gf=r["hg"] if side == "home" else r["ag"],
                 ga=r["ag"] if side == "home" else r["hg"])
        for k, (hc, ac) in stats.items():
            d[k] = r[hc] if side == "home" else r[ac]
            d[k + "_against"] = r[ac] if side == "home" else r[hc]
        d["in_own_country"] = int(HOST_COUNTRY.get(d["team"]) == r.venue_country)
        rows.append(d)
long = pd.DataFrame(rows)


def shrink(sum_prior, n_prior, league_mean):
    return (sum_prior + K * league_mean) / (n_prior + K)


def add_prior_features(long):
    """Rolling pre-match averages using only matches with an EARLIER kick-off."""
    cols = ["gf", "ga", "sot", "poss", "shots", "corners", "intercept",
            "shots_against"]
    out = []
    for i, r in long.iterrows():
        earlier = long[long.kickoff < r.kickoff]
        feat = {}
        for c in cols:
            fb_key = "shots" if c == "shots_against" else c
            fb = FALLBACK[fb_key] if c not in ("gf", "ga") else FALLBACK["gf"]
            league = earlier[c].mean() if len(earlier) else fb
            team_prev = earlier[earlier.team == r.team]
            feat[f"{c}_avg"] = shrink(team_prev[c].sum(), len(team_prev), league)
        prev = earlier[earlier.team == r.team]
        feat["n_prior"] = len(prev)
        if len(prev):
            feat["rest_days"] = min((r.date - prev.date.max()).days, REST_CAP)
        else:
            feat["rest_days"] = REST_CAP
        out.append(feat)
    return pd.concat([long.reset_index(drop=True), pd.DataFrame(out)], axis=1)


long = add_prior_features(long)
long["rank"] = long["team"].map(RANK)

# attach opponent's pre-match features
opp = long[["match_id", "team", "rank", "gf_avg", "ga_avg", "sot_avg", "poss_avg",
            "shots_avg", "shots_against_avg", "intercept_avg", "rest_days",
            "in_own_country"]].copy()
opp.columns = ["match_id", "opp"] + ["opp_" + c for c in opp.columns[2:]]
long = long.merge(opp, on=["match_id", "opp"], how="left")

# ------------------------------------------------------------------ 2.1  (104 rows)
home = long[long.side == "home"].sort_values("match_id").reset_index(drop=True)
lr21 = pd.DataFrame({
    "match_id": home.match_id,
    "home_team": home.team,
    "away_team": home.opp,
    "goal_diff": home.gf - home.ga,                          # TARGET (home - away)
    "rank_dif": home.opp_rank - home["rank"],               # + favours home team
    "host_adv_dif": home.in_own_country - home.opp_in_own_country,
    "knockout": home.knockout,
    "rest_days_diff": home.rest_days - home.opp_rest_days,
    "goals_for_avg_diff": (home.gf_avg - home.opp_gf_avg).round(3),
    "goals_against_avg_diff": (home.ga_avg - home.opp_ga_avg).round(3),
    "sot_avg_diff": (home.sot_avg - home.opp_sot_avg).round(3),
    "possession_avg_diff": (home.poss_avg - home.opp_poss_avg).round(3),
})
lr21.to_csv("lr21_match_dataset.csv", index=False, encoding="utf-8-sig")

# ------------------------------------------------------------------ 2.2  (208 rows)
long = long.sort_values(["match_id", "side"], ascending=[True, True]).reset_index(drop=True)
lr22 = pd.DataFrame({
    "match_id": long.match_id,
    "team": long.team,
    "opponent": long.opp,
    "goals_scored": long.gf,                                  # TARGET
    "team_rank": long["rank"],
    "opp_rank": long.opp_rank,
    "knockout": long.knockout,
    "team_in_own_country": long.in_own_country,
    "team_shots_avg": long.shots_avg.round(3),
    "opp_shots_against_avg": long.opp_shots_against_avg.round(3),
    "team_corners_avg": long.corners_avg.round(3),
    "opp_interceptions_avg": long.opp_intercept_avg.round(3),
})
lr22.to_csv("lr22_teammatch_dataset.csv", index=False, encoding="utf-8-sig")

print(lr21.shape, lr22.shape)
