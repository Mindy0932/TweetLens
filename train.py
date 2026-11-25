"""Run: python train.py. All metrics are from a held-out group split."""
from pathlib import Path
import json, hashlib, platform, time, zipfile
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from src.modeling import EvidenceExtractor, LABELS, jaccard

ROOT = Path(__file__).resolve().parent

def run():
    started = time.time()
    out = ROOT/'artifacts'; out.mkdir(exist_ok=True)
    raw = pd.read_csv(ROOT/'data/train.csv')
    clean = raw.dropna(subset=['text','selected_text','sentiment']).copy()
    clean['group'] = clean.text.str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    clean = clean[clean.group.str.len() > 0].copy()
    # Keep group identity across the split, then remove exact text+label repeats.
    before = len(clean)
    clean = clean.drop_duplicates(['group','sentiment']).reset_index(drop=True)
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(clean, groups=clean.group))
    train, valid = clean.iloc[tr].copy(), clean.iloc[va].copy()
    assert set(train.group).isdisjoint(set(valid.group))
    pd.DataFrame({'textID':clean.textID,'split':np.where(clean.index.isin(tr),'train','validation')}).to_csv(out/'split_manifest.csv', index=False)
    vectorizer = TfidfVectorizer(lowercase=True, strip_accents='unicode', ngram_range=(1,2), min_df=2, max_features=35000, sublinear_tf=True)
    x = vectorizer.fit_transform(train.text); xv = vectorizer.transform(valid.text)
    estimators = {'Majority baseline':DummyClassifier(strategy='most_frequent'), 'Logistic regression':LogisticRegression(C=2, max_iter=1000, random_state=42), 'Linear SVM':LinearSVC(C=.8, random_state=42)}
    metrics, fitted = [], {}
    valid['word_count'] = valid.text.str.split().str.len()
    valid['gold_span'] = valid.selected_text
    for name, model in estimators.items():
        model.fit(x, train.sentiment)
        pred = model.predict(xv)
        metrics.append({'model':name, 'accuracy':accuracy_score(valid.sentiment,pred),'macro_f1':f1_score(valid.sentiment,pred,average='macro'), 'report':classification_report(valid.sentiment,pred,output_dict=True,zero_division=0), 'confusion':confusion_matrix(valid.sentiment,pred,labels=LABELS).tolist()})
        fitted[name] = model
        valid[name] = pred
        print(name, metrics[-1]['accuracy'], metrics[-1]['macro_f1'], flush=True)
    # Default product uses LR probabilities; SVM is an experimental comparator.
    classifier = fitted['Logistic regression']
    valid['prediction'] = classifier.predict(xv)
    valid['confidence'] = classifier.predict_proba(xv).max(axis=1)
    extractor = EvidenceExtractor().fit(train)
    valid['oracle_span'] = [extractor.predict_one(t,l) for t,l in zip(valid.text,valid.sentiment)]
    valid['predicted_span'] = [extractor.predict_one(t,l) for t,l in zip(valid.text,valid.prediction)]
    for key, col in [('whole_text_jaccard','text'),('oracle_jaccard','oracle_span'),('pipeline_jaccard','predicted_span')]:
        valid[key] = [jaccard(a,b) for a,b in zip(valid.gold_span,valid[col])]
    span = [{'sentiment':label,'n':len(g),**{col:float(g[col].mean()) for col in ['whole_text_jaccard','oracle_jaccard','pipeline_jaccard']}} for label,g in [('all',valid),*list(valid.groupby('sentiment'))]]
    # Topic discovery is unsupervised, fitted only on training text. Eight clusters
    # are exploratory bins, not verified semantic classes.
    tv = TfidfVectorizer(stop_words='english', min_df=4, max_df=.85, max_features=10000)
    tx=tv.fit_transform(train.text); vx=tv.transform(valid.text)
    reducer=TruncatedSVD(n_components=40, random_state=42).fit(tx)
    latent=normalize(reducer.transform(tx))
    km=KMeans(n_clusters=8, random_state=42, n_init=10)
    km.fit(latent); valid['topic']=km.predict(normalize(reducer.transform(vx)))
    words=tv.get_feature_names_out()
    centers=np.vstack([np.asarray(tx[km.labels_ == i].mean(axis=0)).ravel() for i in range(8)])
    topic_names={str(i): ' / '.join(words[center.argsort()[-3:][::-1]]) for i,center in enumerate(centers)}
    svd=TruncatedSVD(n_components=2,random_state=42).fit(tx)
    xy=svd.transform(vx); valid['x']=xy[:,0];valid['y']=xy[:,1]
    valid['topic_name']=valid.topic.map(lambda v:topic_names[str(v)])
    top_terms={}
    for label in LABELS:
        cx=list(classifier.classes_).index(label)
        weights=classifier.coef_[cx]
        top_terms[label]=[{'term':str(w),'weight':float(s)} for w,s in zip(vectorizer.get_feature_names_out()[weights.argsort()[-15:][::-1]],sorted(weights,reverse=True)[:15])]
    bundle={'vectorizer':vectorizer,'classifier':classifier,'extractor':extractor,'topic_vectorizer':tv,'topic_model':km,'topic_reducer':reducer,'topic_names':topic_names}
    joblib.dump(bundle,out/'models.joblib',compress=3)
    test=pd.read_csv(ROOT/'data/test.csv')
    submission=pd.DataFrame({'textID':test.textID,'selected_text':[extractor.predict_one(t,l) for t,l in zip(test.text,test.sentiment)]})
    submission.to_csv(out/'submission.csv',index=False)
    valid.drop(columns=['group']).to_csv(out/'validation_predictions.csv',index=False)
    stats={'raw_rows':len(raw),'missing_rows_removed':len(raw)-len(raw.dropna(subset=['text','selected_text','sentiment'])),'duplicates_removed':before-len(clean),'clean_rows':len(clean),'train_rows':len(train),'validation_rows':len(valid),'test_rows':len(test),'seed':42,'group_overlap':0,'class_counts':clean.sentiment.value_counts().to_dict(),'train_class_counts':train.sentiment.value_counts().to_dict(),'validation_class_counts':valid.sentiment.value_counts().to_dict(),'source_sha256':hashlib.sha256((ROOT/'data/train.csv').read_bytes()).hexdigest(),'sklearn_version':sklearn.__version__,'python_version':platform.python_version(),'train_seconds':round(time.time()-started,2),'annotation_alignment_skipped':extractor.skipped_}
    results={'stats':stats,'classification':metrics,'span':span,'topic_names':topic_names,'top_terms':top_terms,'default_model':'Logistic regression','svd_variance_explained':float(svd.explained_variance_ratio_.sum())}
    (out/'metrics.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps({'stats':stats,'span':span},indent=2),flush=True)
    return results

if __name__=='__main__':
    run()
