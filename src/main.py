import TeamStatistics
import MatchStatistics
import argparse

def run(args):
	season_data_file_path = "data/raw/regular_season_detailed_results.csv"
	team_statistic_file_path = "data/structured/average_team_stats_for_all_teams.csv"
	season_tournament_file_path = "data/raw/tourney_detailed_results.csv"
	training_data_match_stats_file_path = "data/structured/training_data_match_statistics.csv"
	
	if(args.team_stats):
		print("Creating Team Statistics")
		TeamStatistics.create(season_data_file_path,team_statistic_file_path)

	if(args.match_stats):
		print("Creating Training Data: Match Statistics")
		MatchStatistics.create(season_data_file_path,season_tournament_file_path,team_statistic_file_path,training_data_match_stats_file_path)


parser = argparse.ArgumentParser(description='Selectively perform dataset creation/training/prediction.')
parser.add_argument('--team-stats', dest='team_stats',action='store_true')
parser.add_argument('--match-stats', dest='match_stats',action='store_true')
run(parser.parse_args());