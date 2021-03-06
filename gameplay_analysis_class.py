#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 10:39:10 2018

@author: brandon
Techniques used:
    - groupby
    - weekday name from date
    - rank
    - sort
    - cumsum (with and without groupby)
    - forward fill (using groupby)
    - week start date method (using ffill with filtering)
    - date subtraction by days (using timedelta)
    - count unique values with a groupby (using nunique())
    - convert date to month start date using numpy
"""
import json
from itertools import product
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, date
# import matplotlib.pyplot as plt
# TODO: have line graphs return png object for web publishing
# TODO: in the summary at the top, have messages explaining movement in ranks


class GamePlayData:
    '''
    Contains data for time spent playing video games, and functions to analyze
    them.
    '''
    def check_completed_list(self):
        last_played = self.last_played()
        all_games = last_played[['title']]
        completed = self._completed
        completed_games = completed[['title', 'complete']]
        merge_games = all_games.merge(completed_games, how='left', on='title')
        missing_games = merge_games[(merge_games['complete'].isnull())]
        return missing_games

    def game_of_the_week(self, num_weeks=16):
        '''
        Shows each week's longest-played game and the hours spent playing for
        that week

        Parameters
        ----------
        num_weeks : int
            This determines the number of weeks to display, beginning from the
            current week and counting backwards.  Defaults to 16.

        Returns
        -------
        weekly_top_games : DataFrame
            The contents DataFrame are displayed graphically.  Fields below.

            week_start : datetime64[ns]
                week-start date (Monday-based) for top game of that week
            hours_played : float64
                Total hours played per game, per week
            title : object (str)
                Name of game played
        '''
        weekly_game_hours = (self._source_data.groupby(['week_start', 'title'],
                                                       as_index=False)
                             [['hours_played']].sum())
        weekly_top_hours = (weekly_game_hours.groupby(['week_start'],
                                                      as_index=False).max()
                            [['week_start', 'hours_played']])
        # TODO: see if there's a more efficient way of doing this
        weekly_top_games = (weekly_top_hours
                            .merge(weekly_game_hours
                                   [['week_start', 'hours_played', 'title']],
                                   on=['week_start', 'hours_played'],
                                   how='left'))
        graph = (weekly_top_games[['title', 'hours_played']].set_index('title')
                 .tail(num_weeks))
        graph.plot(kind='barh')
        most_recent_week = weekly_top_games['week_start'].dt.date.iloc[-1]
        curr_top_game = weekly_top_games['title'].iloc[-1]
        print(f'Top Game for week of {most_recent_week}: {curr_top_game}')
        return weekly_top_games

    def line_weekly_hours(self):
        '''
        Graphs total weekly hours on a line graph
        '''
        graph = self.__agg_week()[['week_start', 'hours_played',
                                   'avg_hrs_per_day']]
        graph = graph.set_index('week_start')
        graph.plot()

    def bar_graph_top(self, num_games=10):
        '''
        This function will create a horizontal bar chart representing total
        time spent playing individual games.
            - Time is ranked with longest at the bottom

        Parameters
        ----------
        num_games : int
            This determines the number of games to display, starting from the
            top.  Defaults to 10.
        '''
        # set data
        top = self.__current_top(self.__agg_total_time_played(), num_games)

        n_groups = top['title'].count()
        game_rank = top['hours_played']
        tick_names = top['title']

        # create plots
        fig, ax = plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.35
        opacity = 0.8

        plt.barh(index + bar_width, game_rank, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Game')

        plt.ylabel('Game')
        plt.xlabel('Hours Played')
        plt.title('Rank by Game Time')
        plt.yticks(index + bar_width, tick_names)

        plt.tight_layout()
        plt.savefig((self._config['output'] + 'games.png'),
                    dpi=300)
        plt.show()

    def pie_graph_top(self, num_games=10):
        '''
        This creates a pie plot of the number of games specified
            - Focuses on overall time spent

        Parameters
        ----------
        num_games : int
            This determines the number of games to display, starting from the
            top.  Defaults to 10.
        '''
        # pop out the first slice (assuming 10 items)
        # explode = (.1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        # plt.pie(game_rank, labels=tick_names, explode=explode)
        top = self.__current_top(self.__agg_total_time_played(), num_games)
        game_rank = top['hours_played']
        tick_names = top['title']
        plt.pie(game_rank, labels=tick_names)
        plt.axis('equal')
        plt.title('Top ' + str(num_games) + ' Distro')
        plt.show()

    def line_graph_top(self, rank_num=5):
        '''
        This graph shows every game that cracked the top rank number specified

        Parameters
        ----------
        rank_num : int
            Defines the range of rank numbers achieved, from 1 to rank_num.
        '''
        rank = self.__agg_time_rank_by_day()
        top = (rank[rank['rank_by_day'] <= rank_num])
        top_list = top['title']
        time_rank_graph = (self.__agg_time_rank_by_day()
                           [self.__agg_time_rank_by_day()['title']
                            .isin(top_list)]
                           [['date', 'title', 'rank_by_day']])

        fig, ax = plt.subplots(figsize=(10, 7))
        plt.ylabel('Rank')
        plt.title('All Games That Reached Top ' + str(rank_num))
        (time_rank_graph.groupby(['date', 'title']).sum()['rank_by_day']
         .unstack().plot(ax=ax))

    def line_graph_curr_top(self, num_games=10):
        '''
        Creates a line graph for current game rankings
            - x-axis is date, y-axis is rank
            - lines represent rank by day

        Parameters
        ----------
        num_games : int
            Similar to rank, this represents the number of games currently
            ranked from 1 to num_games.
        '''
        # create a line graph for game rankings based on num_games value
        top = self.__current_top(self.__agg_total_time_played(), num_games)
        top_list = top['title']
        time_rank_graph = (self.__agg_time_rank_by_day()
                           [self.__agg_time_rank_by_day()['title']
                            .isin(top_list)]
                           [['date', 'title', 'rank_by_day']])

        fig, ax = plt.subplots(figsize=(10, 7))
        plt.ylabel('Rank')
        plt.title('Road to Top ' + str(num_games))
        (time_rank_graph.groupby(['date', 'title']).sum()['rank_by_day']
         .unstack().plot(ax=ax))

    def last_played(self):
        '''
        Calculates the last played date of each game

        Returns
        -------
        last_played_by_game : DataFrame
            Summary of last played dates by game.  Fields below.

            title : object (str)
                Title of game
            date : datetime64[ns]
                Date game was last played
            dow : object (str)
                Day of week game was last played
        '''
        last_played_by_game = (self._source_data.groupby('title',
                                                         as_index=False)
                               .max()[['title', 'date']])
        last_played_by_game = last_played_by_game.sort_values('date')
        last_played_by_game['dow'] = (last_played_by_game['date']
                                      .dt.weekday_name)
        return last_played_by_game

    def log_first_played(self):
        '''
        Logs the first date each game was played

        Returns
        -------
        first_played : DataFrame
            Summary of first played dates by game.  Fields below.

            title : object (str)
                Title of game
            date : datetime64[ns]
                Date game was first played
            dow : object (str)
                Day of week game was first played
        '''
        first_played = (self._source_data
                        .groupby('title', as_index=False)[['date']].min())
        first_played['dow'] = first_played['date'].dt.weekday_name
        first_played.sort_values('date', inplace=True)
        return first_played

    def log_interested_in(self):
        '''
        This looks at games that have been played between ~30 minutes to an
        hour
            - Basically, they haven't been given much time, but there's
              interest

        Returns
        -------
        interested_in : DataFrame
            Summary of games you may be interested in.  Fields below.

            title : object (str)
                Title of game
            minutes_played : int64
                Cumulative minutes played for title
            hours_played : float64
                Cumulative hours played for title
            rank : float64
                Overall rank by time played
        '''
        # find out what games you'd be interested in playing longer
        # use 25% - 50% quartiles to find
        interested_in = pd.DataFrame(self.__agg_total_time_played()
                                     [(self.__agg_total_time_played()
                                       ['minutes_played'] >= 25) &
                                      (self.__agg_total_time_played()
                                       ['minutes_played'] <= 65)])
        return interested_in

    def need_to_play(self, num_games=5):
        '''
        This returns a list of games that have not been completed and have
        not been played in the last 60 days.  Prints titles to console, with
        a heading "Consider playing:"

        Returns
        -------
        been_a_while : DataFrame
            Summary of games that haven't been played recently.  Fields below.

            title : object (str)
                Title of game
            date : datetime64[ns]
                Date last played
            dow : object (str)
                Day of week last played
            minutes_played : int64
                Cumulative minutes played for title
            hours_played : float64
                Cumulative hours played for title
            rank : float64
                Overall rank by time played
            complete : bool
                Indicates whether game has been completed (true) or not (false)
        '''
        last_played = self.last_played()
        hrs_played = self.__agg_total_time_played()
        # merge frames (inner join fine, both frames have the same 1-1 titles)
        been_a_while = last_played.merge(hrs_played, on='title')
        # narrow the list to games played longer than 2 months ago
        current_date = datetime.now()
        two_months_back = current_date - pd.Timedelta(days=60)
        been_a_while = been_a_while[been_a_while['date'] <= two_months_back]
        # check if game has been completed
        been_a_while = been_a_while.merge(self._completed, on='title')
        been_a_while = been_a_while[been_a_while['complete'] == False]
        been_a_while = been_a_while.sort_values(by='rank').head(num_games)
        print('Consider playing:')
        title_list = been_a_while['title'].tolist()
        print("\n".join(title_list))
        return been_a_while

    def graph_single_game_weekly(self, game_title='Octopath Traveler'):
        '''
        Graphs the amount of time spent in hours on a single game by week

        Returns
        -------
        graph_data : DataFrame
            Summary of time spent in hours on a single game.  Fields below.

            index : datetime64[ns]
                Date game was played
            hours_played : float64
                Hours spent playing game for specified date
        '''
        source_data = self.__weekly_hours_by_game()
        source_data = source_data[(source_data['title'] == game_title)]
        graph_data = (source_data[['week_start', 'hours_played']]
                      .set_index('week_start'))
        # add in empty weeks
        start = graph_data.index.min()
        end = graph_data.index.max()
        d = pd.date_range(start, end)
        d = d[(d.weekday_name == 'Monday')]
        d = pd.DataFrame(d).set_index(0)
        graph_data = d.merge(graph_data, left_index=True, right_index=True,
                             how='left')
        graph_data.plot(title=game_title + ' Weekly History')
        return graph_data

    def graph_two_games_weekly(self,
                               game_title_1='Octopath Traveler',
                               game_title_2='Monster Hunter Generations'):
        '''
        Graphs the amount of time spent in hours for two games by week

        Returns
        -------
        graph_data : DataFrame
            Summary of time spent in hours on two games.  Fields below.

            index : datetime64[ns]
                Date game was played
            game_title_1 : float64
                Hours spent playing game for specified date
            game_title_2 : float64
                Hours spent playing game for specified date
        '''
        # TODO: modify this to accept a list, and remove single_game_weekly
        source_data = self.__weekly_hours_by_game()
        # pull data for game_title_1
        source_data_1 = source_data[(source_data['title'] == game_title_1)]
        graph_data_1 = (source_data_1[['week_start', 'hours_played']]
                        .set_index('week_start'))

        # pull data for game_title_2
        source_data_2 = source_data[(source_data['title'] == game_title_2)]
        graph_data_2 = (source_data_2[['week_start', 'hours_played']]
                        .set_index('week_start'))

        # combine data
        graph_data = graph_data_1.merge(graph_data_2, how='outer',
                                        left_index=True, right_index=True)

        # add in empty weeks
        start = graph_data.index.min()
        end = graph_data.index.max()
        d = pd.date_range(start, end)
        d = d[(d.weekday_name == 'Monday')]
        d = pd.DataFrame(d).set_index(0)
        graph_data = d.merge(graph_data, left_index=True, right_index=True,
                             how='left')
        graph_data.columns = [game_title_1, game_title_2]
        graph_data.plot(title='Weekly History')
        return graph_data

    def save_data(self):
        '''
        Outputs data frames to a multiple-tabbed excel file.  Output is
        determined by config.json's "output" string.
        '''
        # run all class methods to gather data for output
        monthly_hour_agg = self.__agg_month()
        weekly_hour_agg = self.__agg_week()
        time_rank_by_day_all = self.__agg_time_rank_by_day()
        total_time_played = self.__agg_total_time_played()
        game_last_played = self.last_played()
        day_of_week_avg = self.__agg_day_of_week()
        game_day_count, total_game_day_count = self.__total_game_day_count()
        date_first_played = self.log_first_played()
        may_be_interested_in = self.log_interested_in()

        # output data
        writer = pd.ExcelWriter((self._config['output'] + 'games.xlsx'))
        total_time_played.to_excel(writer, sheet_name='total_time',
                                   index=False)
        may_be_interested_in.to_excel(writer,
                                      sheet_name='may_be_interested_in',
                                      index=False)
        day_of_week_avg.to_excel(writer, sheet_name='day_of_week_avg',
                                 index=False)
        game_day_count.to_excel(writer, sheet_name='game_day_of_week_count',
                                index=False)
        total_game_day_count.to_excel(writer,
                                      sheet_name='total_game_day_count',
                                      index=False)
        date_first_played.to_excel(writer, sheet_name='date_first_played',
                                   index=False)
        time_rank_by_day_all.to_excel(writer, sheet_name='rank_by_day',
                                      index=False)
        weekly_hour_agg.to_excel(writer, sheet_name='weekly_hours',
                                 index=False)
        game_last_played.to_excel(writer, sheet_name='game_last_played',
                                  index=False)
        monthly_hour_agg.to_excel(writer, sheet_name='monthly_rank',
                                  index=False)
        writer.save()

    def get_streaks(self):
        # data needed: game title and date
        df = self._source_data[['title', 'date']]
        # order by game title first, then date
        df = df.sort_values(['title', 'date'])
        # use groupby with title, shift date down by one to get next day played
        df['next_day'] = df.groupby('title')['date'].shift(-1)
        # fill nat values in next_day with date values (for subtraction)
        df['next_day'] = df['next_day'].fillna(df['date'])
        # subtract number of days between date and next_day, store in column
        df['consecutive'] = (df['next_day'] - df['date'])
        # if column value = 1 day, streak = true (new column)
        df = df[(df['consecutive'] == timedelta(days=1))]
        # need to group streaks
        # test: date - date.shift(-1) = 1 day, and same for next_day
        df['group'] = (((df['next_day'] - df['next_day'].shift(1))
                        == timedelta(days=1)) &
                       ((df['date'] - df['date'].shift(1))
                        == timedelta(days=1)))
        # false represents the beginning of each streak, so it equals 1
        df['streak_num'] = np.where(df['group'] == False, 1, np.nan)
        # forward fill streak_num to complete streak count
        for col in df.columns:
            g = df['streak_num'].notnull().cumsum()
            df['streak_num'] = (df['streak_num'].fillna(method='ffill') +
                                df['streak_num'].groupby(g).cumcount())
        return df

    def check_streaks(self, top_games=10):
        '''
        Evaluates which games have been played at least two consecutive days
        in a row.  Prints any games that are currently being played in a streak
        and graphs the longest streaks along with the days for each
        title.

        Parameters
        ----------
        top_games : int
            Number of games to display in graph.  Graph starts from the game
            with the most consecutive days.

        Returns
        -------
        max_streak : DataFrame
            Superset of data used in graph displaying all games that have been
            played in a streak, along with the number of consecutive days.
            Fields below.

            index : object (str)
                Date game was played
            days : float64
                Days played in a continuous streak

        '''
        df = self.get_streaks()
        # calculate current streak (streaks with yesterday's date)
        # filter down to results with yesterday's date
        yesterday = pd.Timestamp(date.today() - timedelta(1))
        current_streak = (df[(df['next_day'] == yesterday)][['title',
                                                             'streak_num']])
        # turn the title(s) into a list to print
        # if current_streak empty, put 'None' in list
        # TODO: see if printout would look nicer with two lists zipped together
        print('Current streak:')
        if len(current_streak) == 0:
            print('None')
        else:
            current_streak['days'] = current_streak['streak_num'] + 1
            current_streak = (current_streak[['title', 'days']]
                              .set_index('title'))
            print(current_streak)
        # take max streak_sum, grouped by title, store in new object
        max_streak = df.groupby('title', as_index=False)['streak_num'].max()
        max_streak['days'] = max_streak['streak_num'] + 1
        max_streak = max_streak[['title', 'days']].sort_values(['days'],
                                                               ascending=False)
        # graph data
        max_streak = max_streak.reset_index().set_index('title')[['days']]
        chart_title = 'Top ' + str(top_games) + ' Streaks'
        max_streak.head(top_games).plot.barh(title=chart_title)
        return df

    def get_source_data(self):
        '''
        Returns
        -------
        _source_data : DataFrame
            Data Frame of original gameplay data.  Fields below.

            title : object (str)
                Title of game
            time_played : datetime64[ns]
                Time played for given date
            date : datetime64[ns]
                Date game was played
            system : object (str)
                Video game system of title
            minutes_played : int64
                Minutes played for given date
            hours_played : float64
                Hours played for given date
            dow : object (str)
                Day of week that game was played
            cum_total_minutes : int64
                Cumulative minutes played for a given game, running from start
                of data tracking
            week_start : datetime64[ns]
                Date of week start for given date (Monday-based)
            month : datetime64[ns]
                First of the month for given date
        '''
        return self._source_data

    def game_completed(self, game_title):
        # import pdb; pdb.set_trace()
        df = self._completed[self._completed['title'] == game_title]
        game_complete = df['complete'].values[0]
        if game_complete:
            # removes time from date
            date_complete = str(df['date_completed'].values[0])[:10]
            print(game_title + ' was completed on ' + date_complete)
        else:
            print(game_title + ' has not been completed yet.')

    def single_game_history(self, game_title):
        df = self._source_data[self._source_data['title'] == game_title]
        # add total hours spent playing game
        total_hours = df['hours_played'].sum()
        print('Played for ' + str("{0:.2f}".format(total_hours)) + ' hours.')
        # create date range for graph
        # make range start from the 1st of the month on the min side
        min_date = df['date'].min().strftime('%Y-%m-01')
        date_range = pd.DataFrame(pd.date_range(min_date, df['date'].max()))
        date_range.columns = ['date']
        # format date shorter for graph (ax object?)
        # plt.locator_params(axis='x', nbins=10)
        df = pd.merge(df, date_range, how='right',
                      on='date').sort_values('date').reset_index()
        # get positions of start of each month, name/year of month
        locs = df[df['date'].dt.day == 1].index
        # remove time from datetime
        labels = (df[df['date'].dt.day == 1][['date']].values
                  .astype('datetime64[D]').astype('str'))
        df = df[['date', 'hours_played']].set_index('date')
        ax = df.plot.bar(title='Hours Played')
        plt.xticks(locs, labels)
        ax.set_xlabel('date played')
        ax.set_ylabel('hours')
        # df.plot.bar(ax=ax)
        return df

    def single_game_streaks(self, game_title):
        '''
        Gives detailed information on gameplay streaks for specified game_title
        '''
        df = self.get_streaks()
        # import pdb; pdb.set_trace()
        df = df[df['title'] == game_title]
        # check for 0 streaks to avoid errors
        if len(df) != 0:
            # init this to true for first loop
            first_streak = True
            streak_ranges = pd.DataFrame(columns=['start', 'end'])
            # loop through streak_num col, starting with 1 until next 1 reached
            # for index, row in df.iterrows():
            for i, (index, row) in enumerate(df.iterrows()):
                # import pdb; pdb.set_trace()
                # record the date at 1, and also last next_day before next 1
                # two things would trigger logging next_day:
                # 1. we hit streak_num = 1 after first streak_num = 1
                # 2. we hit the end of the dataframe
                start = row['date']
                streak = row['streak_num']
                if streak == 1.0:
                    # need to find out if this is the first streak for logic
                    if first_streak is False:
                        last_df = end
                        # append date at 1 and last next_day to dataframe
                        add_row = pd.DataFrame([(start_df, last_df)],
                                               columns=['start', 'end'])
                        streak_ranges = pd.concat([streak_ranges, add_row])
                    # no matter what, start begins here
                    start_df = start
                    # first_streak = False
                # repeat until end of dataframe is reached
                # check for end of df
                if i == len(df) - 1:
                    last_df = row['next_day']
                    add_row = pd.DataFrame([(start_df, last_df)],
                                           columns=['start', 'end'])
                    streak_ranges = pd.concat([streak_ranges, add_row])
                first_streak = False
                # because of the algorithm's lag, this needs to be logged last
                end = row['next_day']
            # create column for number of days for each streak
            streak_ranges['days'] = (streak_ranges['end']
                                     - streak_ranges['start'])
            # create column for rank based on days for each streak
            streak_ranges['rank'] = streak_ranges['days'].rank(ascending=False,
                                                               method='dense')
            max_days = (streak_ranges[streak_ranges['rank'] == 1][['days']]
                        .values)
            max_start = (streak_ranges[streak_ranges['rank'] == 1][['start']]
                         .values)
            max_end = streak_ranges[streak_ranges['rank'] == 1][['end']].values
            print(str(len(streak_ranges)) + ' streak(s).')
            # fix print out summary of streaks - maximum, total num, etc
            max_days = int(max_days[0][0] / np.timedelta64(1, 'D'))
            max_start = (pd.to_datetime(str(max_start[0][0]))
                         .strftime('%m-%d-%Y'))
            max_end = (pd.to_datetime(str(max_end[0][0]))
                         .strftime('%m-%d-%Y'))
            # TODO: modify to display current streaks if they are longest
            print(f'The longest streak played was for {max_days} days, '
                  f'starting on {max_start} and running until {max_end}.')

            # create graph of streaks and display
            streak_dates = pd.Series()
            for i, (index, row) in enumerate(streak_ranges.iterrows()):
                # start by create date series between each start and end
                start = row['start']
                end = row['end']
                new_range = pd.date_range(start, end)
                streak_dates = streak_dates.append(new_range.to_series())

            graph_data = pd.DataFrame(streak_dates)
            # create value for graphing, and remove extra date column
            graph_data['played'] = 1
            graph_data = graph_data['played']
            # create dates for gaps between streaks
            all_days = (pd.DataFrame(pd.date_range(start=streak_dates.min(),
                                                   end=streak_dates.max()))
                        .set_index(0))
            # join on index with graph_data
            graph_data_final = all_days.join(graph_data, how='left')
            graph_data_final.plot(title='Gameplay Streaks')
        else:
            print('No streaks.')
        return

    def weekly_hours_snapshot(self):
        df = self.__weekly_hours_by_game()
        # filter to current week
        current_week = df['week_start'].max()
        df = (df[df['week_start'] == current_week]
              .set_index('title')[['hours_played']])
        # set_ylabel drops label from the left side of the chart
        df.plot.pie(y='hours_played', figsize=(5, 5),
                    title='Weekly Hours Distribution',
                    legend=False, autopct='%1.1f%%').set_ylabel('')
        return

    def __init__(self):
        '''
        Set up paths, imports data and perform initial calculations:
            - Minutes played
            - Hours played
            - Day of week
            - cumulative minutes (total and by game)
            - Week start date
            - Month
        '''
        with open('config.json') as f:
            config = json.load(f)
        path = (config['input'])
        source = pd.read_csv((path + 'game_log.csv'),
                             parse_dates=['date', 'time_played'])
        complete = pd.read_csv(path + 'completed.csv',
                               parse_dates=['date_completed'])

        # perform initial calculations
        source['minutes_played'] = ((source['time_played'].dt.hour * 60)
                                    + source['time_played'].dt.minute)
        source['hours_played'] = source['minutes_played'] / 60
        source['dow'] = source['date'].dt.weekday_name
        source.sort_values('date', inplace=True)
        source['cum_total_minutes'] = (source.groupby('title')
                                       ['minutes_played'].cumsum())
        # add in start week (Sunday - Sat)
        # subtract the day number from the date (with 0 being Monday)
        source['week_start'] = (source['date'] - pd.to_timedelta
                                (source['date'].dt.dayofweek, unit='d'))
        source['month'] = source['date'].values.astype('datetime64[M]')
        self._source_data = source
        self._completed = complete
        self._config = config

    def __agg_month(self):
        '''
        Roll up monthly totals, returns DataFrame

        Returns
        -------
        monthly_hour : DataFrame
            Displays hours played by month, by title.  Fields below.

            month : datetime64[ns]
                First of the month for given date
            title : object (str)
                Title of game
            hours_played : float64
                Hours played for given month
            rank_by_month : float64
                Title's time ranked for given month
        '''
        monthly_hour = (self._source_data
                        .groupby(['month', 'title'], as_index=False)
                        .sum()[['month', 'title', 'hours_played']])
        monthly_hour['rank_by_month'] = (monthly_hour.
                                         groupby('month')['hours_played'].
                                         rank(method='dense', ascending=False))
        monthly_hour = monthly_hour.sort_values(['month', 'rank_by_month'])
        return monthly_hour

    def __agg_week(self):
        '''
        Creates a DataFame that rolls up weekly totals.  Does not include game
        title information.

        Returns
        -------
        weekly_hour_days : DataFrame
            week_start : datetime64[ns]
                The date of the first Sunday of each week
            hours_played : float64
                Total hours played for the week
            days_sampled : int64
                Number of days containing data for the week
            avg_hrs_per_day : float64
                hours_played / days_sampled
        '''
        weekly_hour = (self._source_data.groupby('week_start', as_index=False)
                       .sum()[['week_start', 'hours_played']])
        weekly_days = (self._source_data.groupby('week_start').nunique()
                       [['date']].reset_index())
        weekly_hour_days = pd.merge(weekly_hour, weekly_days, on='week_start')
        weekly_hour_days.columns = ['week_start',
                                    'hours_played',
                                    'days_sampled']
        weekly_hour_days['avg_hrs_per_day'] = (weekly_hour_days['hours_played']
                                               /
                                               weekly_hour_days['days_sampled']
                                               )
        return weekly_hour_days

    def __agg_time_rank_by_day(self):
        '''
        Ranks total game hours by game, by day
            - Calculates cumulative time spent in each game by day
            - Ranks each game by cumulative time for each day

        Returns
        -------
        time_rank : DataFrame
            date : datetime64[ns]
                Date of data sample.  This does not indicate that the game was
                played, it represents cumulative totals for the game at this
                date.
            title : object (str)
                Title of game
            cum_total_minutes : float64
                Cumulative minutes played for a given game, running from start
                of data tracking
            rank_by_day : float64
                Title's time ranked for given day
        '''
        time_rank_by_day = pd.DataFrame(self
                                        ._source_data[['title', 'date',
                                                       'cum_total_minutes']])
        game_list = pd.Series(self._source_data['title'].unique())
        date_list = pd.Series(self._source_data['date'].unique())

        date_game = pd.DataFrame(list(product(date_list, game_list)),
                                 columns=['date', 'title'])
        time_rank = pd.DataFrame(date_game.merge(time_rank_by_day, how='left'))
        time_rank['rank_by_day'] = (time_rank.groupby('date')
                                    ['cum_total_minutes']
                                    .rank(method='dense', ascending=False))
        time_rank.sort_values(['title', 'date'], inplace=True)

        time_rank['cum_total_minutes'] = (time_rank.groupby('title')
                                          ['cum_total_minutes'].ffill())

        time_rank = time_rank[time_rank['cum_total_minutes'].notnull()]

        time_rank['rank_by_day'] = (time_rank.groupby('date')
                                    ['cum_total_minutes']
                                    .rank(method='dense', ascending=False))

        time_rank = time_rank.sort_values(['date', 'rank_by_day'])
        return time_rank

    def __agg_time_rank_by_week(self):
        '''
        Ranks total game hours by game, by week
            - Calculates time spent in each game by week
            - Ranks each game by time for each day

        Returns
        -------
        time_rank : DataFrame
            week_start : datetime
                The date of the first Sunday of each week
            title : object
                The title of the game played
            minutes_played : float
                Total minutes played in the week by game
            rank_by_week : float
                Rank based on total time played for the week, with 1
                representing the most time
        '''

        time_rank_by_week = pd.DataFrame(self
                                         ._source_data.groupby(['title',
                                                                'week_start'],
                                                               as_index=False)
                                         .sum())
        game_list = pd.Series(self._source_data['title'].unique())
        week_list = pd.Series(self._source_data['week_start'].unique())

        week_game = pd.DataFrame(list(product(week_list, game_list)),
                                 columns=['week_start', 'title'])
        time_rank = pd.DataFrame(week_game.merge(time_rank_by_week,
                                                 how='left'))
        time_rank['rank_by_week'] = (time_rank.groupby('week_start')
                                     ['minutes_played']
                                     .rank(method='dense', ascending=False))
        time_rank.sort_values(['week_start', 'rank_by_week'], inplace=True)
        time_rank = time_rank[time_rank['minutes_played'].notnull()]
        time_rank = time_rank[['week_start', 'title', 'minutes_played',
                               'rank_by_week']]

        return time_rank

    def __agg_total_time_played(self):
        '''
        Sums up total time played by game, and ranks totals

        Returns
        -------
        time_played : DataFrame
            title : object (str)
                Title of game
            minutes_played : int64
                Total minutes spent playing the game since data tracking began
            hours_played : float64
                Total hours spent playing the game since data tracking began
            rank : float64
                Rank of game based on time spent playing, with 1 representing
                the most time spent
        '''
        time_played = pd.DataFrame(self._source_data.groupby('title',
                                                             as_index=False)
                                   [['title',
                                     'minutes_played',
                                     'hours_played']].sum())
        time_played.sort_values('minutes_played', ascending=False,
                                inplace=True)
        time_played['rank'] = (time_played['minutes_played']
                               .rank(method='dense', ascending=False))
        return time_played

    def __weekly_hours_by_game(self):
        '''
        Shows total hours played for each game for each week

        Returns
        -------
        weekly_game_hours : DataFrame
            week_start : datetime64[ns]
                Week-start date (Monday-based) for title played
            title : object (str)
                Title of game
            hours_played : float64[ns]
                Total hours spent playing title for the week
        '''
        weekly_game_hours = (self._source_data.groupby(['week_start', 'title'],
                                                       as_index=False)
                             [['hours_played']].sum().sort_values(
                                                      ['week_start',
                                                       'hours_played'],
                                                      ascending=False))
        return weekly_game_hours

    def __current_top(self, source_df, rank_num):
        '''
        This generates data for the specified number of top ranked games
            - Primarily used to limit data points in graphs

        Returns
        -------
        top : DataFrame
            title : object (str)
                Title of game
            minutes_played : int64
                Total minutes spent playing the game since data tracking began
            hours_played : float64
                Total hours spent playing the game since data tracking began
            rank : float64
                Overall rank by time played
        '''
        top = source_df[source_df['rank'] <= rank_num]
        return top

    def __agg_day_of_week(self):
        '''
        Creates a DataFrame containing the average playtime per day of week

        Returns
        -------
        day_of_week : DataFrame
            dow : object (str)
                Day of week
            minutes_played : float64
                Minutes played by day of week
            hours_played : float64
                Hours played by day of week
            rank : float64
                Rank for each day of week, with 1 representing the most time
                spent
        '''
        # day of week patterns
        day_of_week = (self._source_data.groupby('dow', as_index=False)
                       [['dow', 'minutes_played', 'hours_played']].mean())
        day_of_week['rank'] = (day_of_week['minutes_played']
                               .rank(method='average', ascending=False))
        day_of_week.sort_values('minutes_played', ascending=False,
                                inplace=True)
        return day_of_week

    def __total_game_day_count(self):
        '''
        Runs a count of how many days each game was played
            - This is not 24-hour days, just a count of 1 per each individual
              date

        Returns
        -------
        day_count : DataFrameGroupBy
            dow : object (str)
                Day of week title was played
            title : object (str)
                Title of game
            DaysPlayed : int64
                Number of times game was played on this day
        total_day_count : DataFrameGroupBy
            title : object (str)
                Title of game
            DaysPlayed : int64
                Number of individual days that game was played
            rank : int64
                Rank based on number of individual days game was played
        '''
        day_count = (self._source_data.groupby(['dow', 'title'],
                                               as_index=False)
                     [['time_played']].count())
        day_count.columns = ['dow', 'title', 'DaysPlayed']

        total_day_count = (day_count.groupby('title', as_index=False)
                           [['DaysPlayed']].sum())
        total_day_count['rank'] = (total_day_count['DaysPlayed']
                                   .rank(ascending=False, method='dense'))
        total_day_count.sort_values('rank', inplace=True)
        return day_count, total_day_count
# %%


def summarize_all():
    '''
    This outputs the dashboard graphs to the console and saves export data
    '''
    game_data = GamePlayData()
    game_data.game_of_the_week()
    game_data.weekly_hours_snapshot()
    game_data.need_to_play()
    game_data.check_streaks()
    game_data.line_weekly_hours()
    game_data.bar_graph_top()
    game_data.pie_graph_top()
    game_data.line_graph_top()
    game_data.graph_two_games_weekly()
    # TODO: find number of hours played in each game since it was completed
    game_data.save_data()
# %%


def summarize_game(game_title):
    '''
    Outputs summary information and graphs for specified game_title
    '''
    game_data = GamePlayData()
    game_data.game_completed(game_title)
    # show play history with dates and hours (bar graph)
    game_data.single_game_history(game_title)
    # show game streaks
    game_data.single_game_streaks(game_title)
# %%


# main
summarize_all()
