from pathlib import Path
import pickle as pkl

def load_models():
    folder_path = Path("models")

    all_files = [f.name for f in folder_path.iterdir() if f.is_file()]
    meta = {}
    models = {}
    
    for file in all_files:
        with open(f'models/{file}', 'rb') as f:
            fname = file.split('.')[0]
            if fname.lower() in ('svc', 'logisticregression', 'gaussiannb', 'randomforestclassifier'):
                models[fname] = pkl.load(f)
            else:
                meta[fname] = pkl.load(f)
            
    return models, meta

if __name__ == "__main__":
    models, meta = load_models()

    scaler = meta['scaler']
    eigenvectors = meta['eigenvectors']
    
    import pandas as pd
    
    df = pd.read_csv('test.csv')
    print(len(df.columns))
    
    scaled = scaler.transform(df)
    projection = scaled @ eigenvectors
    print(projection.shape)
    
    print(models['LogisticRegression'].predict(projection))