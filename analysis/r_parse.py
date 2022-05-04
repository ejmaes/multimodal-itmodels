import pandas as pd
import re, sys, os
from tqdm import tqdm
import argparse

# %%
def parse_r_results(lines:list):
    res = []
    mode = 'out'
    for line in tqdm(lines):
        line = line.replace('\n','')
        if mode == "fe" and (line == "---" or line.strip() == ''):
            mode = "out"
        elif len(line) == 0:
            pass
        elif r"mt <- mta[mta$model" in line: # also a command so need to be checked before
            file = line.split('"')[1] # exactly 3 splits
        elif line[0] == ">" and "summary" not in line:
            command = line
        elif "Error in" in line:
            res.append({
                'file': file,
                'command': command,
                'res': line
            })
        elif "fixed effects:" == line.lower().strip():
            mode = "fe"
        elif "Formula" in line:
            formula = line.split(':')[-1].strip()
        elif mode == "fe":
            #if line[0] == '(':
            if line[0] != ' ':
                line = line.replace('< 2e-16','<2e-16')
                line = [x for x in line.split() if x != '']
                #[vname, est, std, dt, t, p, sig] = line
                res.append({
                    'data_file': file,
                    'which_entropy': 'xu_h' if 'logh' in formula or 'xu_h' in formula else 'normalised_h',
                    'speaker': [y for y in ["'f'","'g'"] if y in command],
                    'level': 'theme' if 'theme_id' in formula or 'logt' in formula else 'interaction',
                    'origin': 'GF' if 'log' in formula else 'XR',
                    'command': command,
                    'formula': formula,
                    'sig': '' if len(line) < 7 else line[6],
                    'var': line[0], 'estimate': line[1], 't': line[4], 'p': line[5], 
                })
            
    return pd.DataFrame(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data args
    parser.add_argument("--data_path", "-i", type=str, required=True, help="Path to R tests output.")
    parser.add_argument("--output_path", "-o", type=str, default=None, help="Path to output the xlsx file to.")
    parser.add_argument("--full_df", "-f", action="store_true", help="Whether to output a full df and not only the selected columns.")
    parser.add_argument("--no_pivot_df", "-np", action="store_true", help="Whether to output a full df and not only the selected columns.")
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = '.'.join(args.data_path.split('.')[:-1]) + '_parse.xlsx'
    d = {'':'', "'f'":' - follower', "'g'": ' - giver'}
    
    # read file
    with open(args.data_path,'r') as f:
        lines = f.readlines()
    res = parse_r_results(lines)

    res['speaker'] = res['speaker'].apply(lambda x: '' if str(x) == 'nan' or len(x) == 0 else x[0])
    
    print("Error lines: \n")
    print(res[~res['res'].isna()])

    df = res[res['res'].isna()]
    df['test_label'] = df.apply(lambda x: f"Position in {x['level'].upper()}{d[x.speaker]}", axis=1)

    if not args.no_pivot_df:
        df = df.set_index(['data_file','command','formula']).dropna(axis=1)
        itc = df[df['var'] == "(Intercept)"][['sig']].rename(columns={'sig':'intercept_sig'})
        df = df[df['var'] != '(Intercept)']
        df = pd.concat([df, itc], axis=1).reset_index()
        int_cols = ['data_file', 'test_label', 'formula', 'origin', 'which_entropy', 'intercept_sig', 'estimate', 'p', 'sig']
    else:
        ['data_file', 'test_label', 'formula', 'origin', 'which_entropy', 'estimate', 'p', 'sig']
    
    if not args.full_df:
        df.sort_values(['data_file','origin','speaker'])[int_cols].to_excel(args.output_path, index=False)
    else:
        df.sort_values(['data_file','origin','speaker']).to_excel(args.output_path, index=False)