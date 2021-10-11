import sqlite3
import pandas as pd

latest_file_timestamp = '20211010-151010'

PATH_INTERACTIONS = f'postgres/{latest_file_timestamp}_grab_cut_interaction.csv'
PATH_MASKS = f'postgres/{latest_file_timestamp}_grab_cut_mask.csv'

IMG_SHAPE = (384, 512)

from datetime import datetime
import time

now = time.time()
offset = datetime.fromtimestamp(now) - datetime.utcfromtimestamp(now)


def utc_to_local(utc_datetime, offset=offset):
    localtime = (utc_datetime + offset).time()
    d = datetime.strptime(localtime, '%Y-%m-%d %H:%M:%S')
    return d.strftime('%Y-%m-%d %H:%M:%S')


def format_time(time):
    return time.strftime('%Y-%m-%d %H:%M:%S')


images = ['lym0', 'lym17', 'neu0', 'neu12', 'neu90', 'agg17', 'agg23', 'agg185', 'neurblas5', 'neurblas10',
          'neurblas16', 'normal2', 'normal4']
difficulties = ['Easy', 'Easy', 'Medium', 'Medium', 'Medium', 'Noisy', 'Noisy', 'Noisy', 'Difficult', 'Difficult',
                'Difficult', 'Difficult', 'Difficult']
true_cell_count = [1, 1, 7, 6, 4, 3, 2, 3, 18, 16, 16, 26, 15]

image_metadata_df = pd.DataFrame.from_dict({
    'image_id': images,
    'difficulty': difficulties,
    'cell_count': true_cell_count,
})

# Read from CSV, drop some rows from testing phase and format times and column names
interactions_df = pd.read_csv(PATH_INTERACTIONS, skiprows=[1, 2, 3, 4, 5, 6, 7, 8])
interactions_df['first_interaction_time'] = pd.to_datetime(interactions_df['first_interaction_time'], utc=True)
interactions_df['submission_time'] = pd.to_datetime(interactions_df['submission_time'], utc=True)
interactions_df = interactions_df.rename(columns={'id': 'interaction_uuid'})

print(f"[*] {len(interactions_df)} segmentation requests have been made.")

interactions_df['first_interaction_time'] = interactions_df['first_interaction_time'] + pd.Timedelta(2, unit='h')
interactions_df['first_interaction_time'] = interactions_df['first_interaction_time'].apply(format_time)

interactions_df['submission_time'] = interactions_df['submission_time'] + pd.Timedelta(2, unit='h')
interactions_df['submission_time'] = interactions_df['submission_time'].apply(format_time)

image_categories = ['lym0', 'lym17', 'neu0', 'neu12', 'neu90', 'agg17', 'agg23', 'agg185', 'neurblas5', 'neurblas10', 'neurblas16', 'normal2', 'normal4']
imgs = pd.api.types.CategoricalDtype(ordered=True, categories=image_categories)

interactions_df['image_id'] = interactions_df['image_id'].astype(imgs)

# Read from CSV, format column name
masks_df = pd.read_csv(PATH_MASKS)
masks_df = masks_df.rename(columns={'id': 'mask_uuid'})
masks_df = masks_df.rename(columns={'interactionrecord_id': 'interaction_uuid'})

print(f"[*] {len(masks_df)} masks have been submitted.")

# Get interactions where there is both a record and a mask available.
interactions_with_submission_df = pd.merge(interactions_df, masks_df, how='inner', on=['session_id', 'image_id'])

interactions_with_submission_df.rename(columns={
    'interaction_uuid_x': 'interaction_uuid',
    'mask_uuid': 'submitted_mask_uuid',
    'mask': 'submitted_mask',
}, inplace=True)
del interactions_with_submission_df['interaction_uuid_y']

print(f'[*] {len(interactions_with_submission_df)} segmentations requested given that a final mask choice was submitted.')

interactions_df = pd.merge(interactions_df, image_metadata_df, how='inner', on=['image_id'])
interactions_with_submission_df = pd.merge(interactions_with_submission_df, image_metadata_df, how='inner', on=['image_id'])

# Write back to db
conn = sqlite3.connect('grabcutstudy.db')
print(f"[*] Connected to DB.")
image_metadata_df.to_sql(name='images', con=conn, index=False, if_exists='replace')
interactions_df.to_sql(name='interactions', con=conn, index=False, if_exists='replace')
masks_df.to_sql(name='masks', con=conn, index=False, if_exists='replace')
interactions_with_submission_df.to_sql(name='interactionswithsubmissions', con=conn, index=False, if_exists='replace')
print(f"[*] Wrote to DB.")
conn.close()
