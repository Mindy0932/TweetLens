"""Build a self-contained, offline HTML dashboard with real model inference."""
from pathlib import Path
import json
import joblib
import pandas as pd
ROOT=Path(__file__).resolve().parent

def build():
    b=joblib.load(ROOT/'artifacts/models.joblib')
    metrics=json.loads((ROOT/'artifacts/metrics.json').read_text())
    df=pd.read_csv(ROOT/'artifacts/validation_predictions.csv')
    v=b['vectorizer'];c=b['classifier']
    fields=['textID','text','sentiment','prediction','confidence','gold_span','predicted_span','pipeline_jaccard','topic']
    payload={'metrics':metrics,'rows':df[fields].to_dict('records'),'points':df.sample(min(1200,len(df)),random_state=42)[['text','sentiment','topic','x','y']].to_dict('records'),'model':{'vocabulary':v.get_feature_names_out().tolist(),'idf':v.idf_.round(9).tolist(),'coef':c.coef_.round(9).tolist(),'intercept':c.intercept_.round(9).tolist(),'classes':c.classes_.tolist()},'extractor':b['extractor'].weights_}
    text=json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    template=(ROOT/'reports/template.html').read_text()
    path=ROOT/'TweetLens_Showcase.html'
    path.write_text(template.replace('__PAYLOAD__',text),encoding='utf-8')
    print(path,round(path.stat().st_size/1024/1024,2),'MiB')
if __name__=='__main__':build()
