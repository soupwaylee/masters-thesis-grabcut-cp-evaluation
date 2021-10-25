import pandas as pd


def prepare_study_access_timeline_df(respondents_with_submissions_df, start_date='2021-09-16', end_date='2021-10-13'):
    padded_timeline_df = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq='D'), columns=['padded_time'])
    respondents_with_submissions_timeline_df = pd.merge(
        respondents_with_submissions_df,
        padded_timeline_df,
        how='outer',
        left_on='participation_time',
        right_on='padded_time'
    )
    respondents_with_submissions_timeline_df.drop('participation_time', axis=1, inplace=True)
    respondents_with_submissions_timeline_df.rename(columns={'padded_time': 'participation_time'}, inplace=True)
    respondents_with_submissions_timeline_df['year'] = respondents_with_submissions_timeline_df['participation_time']\
        .dt.year.astype('int32')
    respondents_with_submissions_timeline_df['month'] = respondents_with_submissions_timeline_df['participation_time']\
        .dt.month.astype('int32')
    respondents_with_submissions_timeline_df['day'] = respondents_with_submissions_timeline_df['participation_time']\
        .dt.day.astype('int32')
    respondents_with_submissions_timeline_df = respondents_with_submissions_timeline_df.fillna("")

    def is_empty_string(x):
        return 0 if x == '' else 1

    respondents_with_submissions_timeline_df['access'] = respondents_with_submissions_timeline_df['session_id']\
        .apply(is_empty_string)

    participation_timeline_grouped_df = respondents_with_submissions_timeline_df.groupby(['year', 'month', 'day'])\
        .sum()\
        .reset_index()
    participation_timeline_grouped_df['date'] = pd.to_datetime(
        participation_timeline_grouped_df[['year', 'month', 'day']]
    ).dt.strftime('%m-%d')

    return participation_timeline_grouped_df
