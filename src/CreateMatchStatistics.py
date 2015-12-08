
# 
# Creates MATCH statistics for all tournament matches in a season (for all seasons)
# 

import pandas as pd
import numpy as np
import re

ROOT_DIR = "../"

def main():
	season_statistics = pd.read_csv(ROOT_DIR+"data/raw/regular_season_detailed_results.csv");
	tournament_statistics = pd.read_csv(ROOT_DIR+"data/raw/tourney_detailed_results.csv");
	team_statistics = pd.read_csv(ROOT_DIR+"data/structured/average_team_stats_for_all_teams.csv");
	all_teams = pd.read_csv(ROOT_DIR+"data/raw/teams.csv")
	
	all_tournament_seasons = np.unique(tournament_statistics.season)

	training_data = pd.DataFrame()

	for season in all_tournament_seasons:
		print("Processing Data for :"+str(season))
		tournament_statistics_for_season = tournament_statistics[tournament_statistics.season == season]
		team_statistics_for_season = team_statistics[team_statistics.season == season]
		for index,row in tournament_statistics_for_season.iterrows():
		    tournament_match = row

		    #Find the two teams
		    winning_team_id = tournament_match['wteam']
		    losing_team_id =  tournament_match['lteam']

		    #Here, we enforce a convention for labeling: 
		    #if: win_team_id  < lose_team_id 
		        #(1101) < (1102)
		        #then: output label: 0 -> 1101 won
		    #else: 
		        #(1102) > (1101)    
		        #then: output label: 1 -> 1102 won

		    #Get their stats_vector from team_stats
		    if(winning_team_id < losing_team_id):
		        team1_statistic = team_statistics_for_season[team_statistics_for_season.teamId == winning_team_id]
		        team2_statistic = team_statistics_for_season[team_statistics_for_season.teamId == losing_team_id]
		        output_label = 0
		    else:
		        team1_statistic = team_statistics_for_season[team_statistics_for_season.teamId == losing_team_id]
		        team2_statistic = team_statistics_for_season[team_statistics_for_season.teamId == winning_team_id]
		        output_label = 1

		    team1_np_vector = team1_statistic[team1_statistic.columns[2:]].as_matrix()
		    team2_np_vector = team2_statistic[team2_statistic.columns[2:]].as_matrix()

		    #Find difference between the vectors
		    match_vector = team1_np_vector - team2_np_vector


		    #Take the difference vector and append it with the label it with 0 if first team won, and 1 if the second team won
		    match_statistics = pd.DataFrame(match_vector,columns=team1_statistic.columns[2:])
		    match_output_label = pd.DataFrame([output_label],columns=["winningTeam"])

		    row_vector_for_training_data = pd.concat([match_statistics,match_output_label],axis=1)
		    row_vector_for_training_data.season = season

		    # Append to training-data-for-season
		    training_data = training_data.append(row_vector_for_training_data,ignore_index=True)		
	training_data.to_csv(ROOT_DIR+"data/structured/training_data_match_statistics.csv")

main()