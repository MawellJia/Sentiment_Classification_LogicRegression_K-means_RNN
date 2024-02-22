import os
import sys
from zipfile import ZipFile


if __name__ == '__main__':
    uid = input('Your UID: ')
    uid.strip()
    fzip = f'{uid.lower()}.zip'

    # check if the answers PDF exists
    possible_filenames = [f'{fname}.{ftype}' for fname in (uid.lower(), uid.upper()) for ftype in ('pdf', 'PDF')]
    pdf = None
    for f in possible_filenames:
        if os.path.exists(f):
            pdf = f
            break

    # if the answers PDF not found
    if pdf is None:
        print(f'ERROR: Could not find your answers PDF "{uid.lower()}.pdf" in the current directory!')
        print(f'Submission ZIP archive NOT created!')
        sys.exit(0)

    # files required to submit
    files_to_submit = [
        pdf,
        'classifier.py',
        'clustering.py',
        'features.py',
        'kmeans.py',
        'rnn_name_generator.py',
        'sentiment_analysis.py',
        'word2vec.py',
    ]

    # sanity check
    for f in files_to_submit:
        assert os.path.exists(f)
    if os.path.exists(fzip):
        fbackup = f'{uid.lower()}_backup.zip'
        print(f'WARN: "{fzip}" already exists, renaming it to {fbackup}!')
        os.rename(fzip, fbackup)

    # create a ZIP archive with all the required files
    with ZipFile(fzip, 'w') as zf:
        for fname in files_to_submit:
            print(f'Adding {fname} ...')
            zf.write(fname)

    print(f'Successfully created submission ZIP archive "{fzip}", please double-check its content before submitting it on Wattle.')

