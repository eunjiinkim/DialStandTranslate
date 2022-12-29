import pandas as pd
from tqdm import tqdm
import re
import argparse
import glob

def process_utt(utt,standard=True):
    if standard:
        idx=1
    else:
        idx=0
    tmp=re.findall('([가-힣]+/[가-힣]+)',utt)
    if tmp:
        for t in tmp:
            utt = re.sub('([가-힣]+/[가-힣]+)',t.split('/')[idx],utt)
    utt=re.sub('{|}|-[가-힣]+-|\(|\)|\(\([가-힣]+\)\)|#|[A-Za-z]|~|\*','',utt)
    utt=re.sub('  ',' ',utt)
    
    return utt

def process_pair(words):
    words = re.sub('{|}|-[가-힣]+-|\(|\)|\(\([가-힣]+\)\)|#|[A-Za-z]|~|\*','',words)
    words = re.sub('  ', ' ', words)
    return words

def process_df(filename):
    print('>> On {}'.format(filename))
    df=pd.read_csv(filename,sep='\t', lineterminator='\n')
    
    print('>> Original data lens: {}'.format(len(df)))
    df.dropna(inplace=True)
    df = df[~df.standard.str.contains('&')]
    df = df[df.standard.apply(lambda x: len(str(x).split())>3)]
    new_df = df.reset_index(drop=True)
    new_df['standard'] = new_df['standard'].map(lambda x: process_utt(x,True))
    new_df['dialect']= new_df['dialect'].map(lambda x: process_utt(x,False))
    new_df['mp_d']= new_df['mp_d'].map(lambda x: process_pair(x))
    new_df['mp_s']= new_df['mp_s'].map(lambda x: process_pair(x))
    
    save_path = filename.split('_data')[0]+'_cleaned.tsv'
    new_df.to_csv(save_path,index=False,sep='\t')
    
    print('>> Processed data lens: {}'.format(len(new_df)))
    print('>> Done & Saved in {}'.format(save_path))
    
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Data Processing')


    parser.add_argument('--region',
                    type=str,
                    default='jeonla',
                    help='dialect region')

    args = parser.parse_args()
    
    files = glob.glob(f'data/{args.region}/*/*.tsv')
    print(files)
    for f in tqdm(files):
        process_df(f)