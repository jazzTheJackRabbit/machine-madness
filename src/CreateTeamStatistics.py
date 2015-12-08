
# 
# Creates TEAM statistics for all tournament matches in a season (for all seasons)
# 

import pandas as pd
import numpy as np
import re

ROOT_DIR = "../"

def main():
	dataset = pd.read_csv(ROOT_DIR+"data/raw/regular_season_detailed_results.csv");
	all_teams = pd.read_csv(ROOT_DIR+"data/raw/teams.csv")

	cols = list(dataset.columns)

	all_seasons = np.unique(dataset.season)
	trainingStatsForAllTeams = pd.DataFrame()
	for year in all_seasons:
		print("Processing for year:"+str(year))	

		for index,row in all_teams.iterrows():
			teamId = all_teams.team_id[index]

			# Stats for target team as winning team
			r = re.compile('^w')
			winningStatColumnNames = filter(r.match, list(dataset.columns))
			winningStatColumnNames.remove('wteam')
			winningStatColumnNames.remove('wloc')
			averageStatsAsWinningTeam = dataset[(dataset.wteam == teamId) & (dataset.season == year)][winningStatColumnNames].mean()
			averageStatsAsWinningTeam = pd.DataFrame(averageStatsAsWinningTeam).transpose()
			averageStatsAsWinningTeam

			# Stats for target team as losing team
			    # Potential change here is to include Average Stats of the teams that lose to this team.
			r = re.compile('^l')
			losingStatColumnNames = filter(r.match, list(dataset.columns))
			losingStatColumnNames.remove('lteam')
			averageStatsAsLosingTeam = (dataset[(dataset.lteam == teamId) & (dataset.season == year)])[losingStatColumnNames].mean()
			averageStatsAsLosingTeam = pd.DataFrame(averageStatsAsLosingTeam).transpose()
			averageStatsAsLosingTeam

			# Meta data
			otherStatsForTargetTeam = pd.DataFrame([[teamId,year]],columns=['teamId','season'])

			trainingStatsForTargetTeam = pd.concat([otherStatsForTargetTeam,averageStatsAsWinningTeam,averageStatsAsLosingTeam],axis=1)

			trainingStatsForAllTeams = trainingStatsForAllTeams.append(trainingStatsForTargetTeam,ignore_index=True)

		trainingStatsForAllTeams.to_csv(ROOT_DIR+"data/structured/average_team_stats_for_all_teams.csv")

main()