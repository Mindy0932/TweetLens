"""Launch with: python -m streamlit run app.py"""
from pathlib import Path
import html, json
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.modeling import LABELS
ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='TweetLens',page_icon='◉',layout='wide')
st.markdown('''<style>.stApp{background:#f5f7fb} h1,h2,h3{color:#152440} [data-testid="stMetric"]{background:white;border:1px solid #dce3ed;border-radius:14px;padding:18px} .stButton>button{background:#2159e8;color:white;border:0;border-radius:9px} </style>''',unsafe_allow_html=True)
@st.cache_resource
def load():
    return joblib.load(ROOT/'artifacts/models.joblib')
@st.cache_data
def data():
    return pd.read_csv(ROOT/'artifacts/validation_predictions.csv'),json.loads((ROOT/'artifacts/metrics.json').read_text())
bundle=load(); df,m=data(); stats=m['stats']

st.title('从一条推文，读懂情感与表达')
st.write('情感分类 · 关键片段提取 · 探索式主题聚类 | 基于 Kaggle Tweet Sentiment Extraction 数据')
a,b,c,d=st.columns(4)
a.metric('清洗后样本',f"{stats['clean_rows']:,}");b.metric('独立验证样本',f"{stats['validation_rows']:,}");c.metric('分类 Macro-F1',f"{m['classification'][1]['macro_f1']:.3f}");d.metric('已知情感时片段 Jaccard',f"{m['span'][0]['oracle_jaccard']:.3f}")
tabs=st.tabs(['数据与主题','模型实验','推文分析','错误检查','方法与来源','RoBERTa 对照'])
colors={'negative':'#ef6b70','neutral':'#8b9bb5','positive':'#16a68c'}
with tabs[0]:
    l,r=st.columns(2)
    with l:
        counts=pd.DataFrame({'情感':list(stats['class_counts']),'数量':list(stats['class_counts'].values())})
        st.plotly_chart(px.bar(counts,x='情感',y='数量',color='情感',color_discrete_map=colors,title='清洗后语料的标签分布'),width='stretch')
    with r:
        st.plotly_chart(px.histogram(df,x='word_count',color='sentiment',barmode='overlay',nbins=30,color_discrete_map=colors,title='验证集文本长度分布'),width='stretch')
    topic=st.selectbox('探索主题簇（自动关键词，未经人工命名）',['全部']+list(m['topic_names'].values()))
    filtered=df if topic=='全部' else df[df.topic_name==topic]
    st.plotly_chart(px.scatter(filtered.sample(min(1200,len(filtered)),random_state=42),x='x',y='y',color='sentiment',color_discrete_map=colors,hover_data=['text','topic_name'],title='TF-IDF 的二维 SVD 投影（距离仅作探索参考）',opacity=.55),width='stretch')
    st.caption(f"二维投影保留约 {m['svd_variance_explained']:.1%} 的方差；不能把二维距离当作完整语义距离。")
    heat=pd.crosstab(df.topic_name,df.sentiment,normalize='index').reindex(columns=LABELS,fill_value=0)
    st.plotly_chart(px.imshow(heat,text_auto='.0%',color_continuous_scale='Blues',title='主题簇内情感构成（验证集真实标签）'),width='stretch')
with tabs[1]:
    table=pd.DataFrame([{k:v for k,v in r.items() if k in ['model','accuracy','macro_f1']} for r in m['classification']])
    st.plotly_chart(px.bar(table.melt(id_vars='model'),x='model',y='value',color='variable',barmode='group',range_y=[0,1],title='相同验证集上的分类模型比较'),width='stretch')
    st.dataframe(table,width='stretch',hide_index=True)
    st.plotly_chart(px.imshow(m['classification'][1]['confusion'],x=LABELS,y=LABELS,text_auto=True,color_continuous_scale='Blues',labels={'x':'预测','y':'真实'},title='逻辑回归混淆矩阵'),width='stretch')
    st.subheader('片段提取：区分已知标签与完整流程')
    st.dataframe(pd.DataFrame(m['span']),width='stretch',hide_index=True)
    st.info('oracle_jaccard 使用数据集真实情感标签；pipeline_jaccard 使用模型预测标签。后者包含分类误差，不是原比赛的同一输入条件。')
    st.caption('Jaccard 评价预测片段和人工标注的词集合重合，不能解释为准确率。所有结果均为本地验证，未提交官方榜单。')
with tabs[2]:
    text=st.text_area('输入英文推文', 'I love the design, but the service was really disappointing.')
    mode=st.radio('情感来源',['模型预测','手动指定（原比赛条件）'],horizontal=True)
    manual=st.selectbox('指定情感',LABELS) if mode.startswith('手动') else None
    if st.button('分析推文',type='primary'):
        if not text.strip():st.warning('请输入文字。')
        else:
            x=bundle['vectorizer'].transform([text]); probs=bundle['classifier'].predict_proba(x)[0]
            prediction=bundle['classifier'].classes_[probs.argmax()]; label=manual or prediction
            span=bundle['extractor'].predict_one(text,label); start=text.find(span)
            st.success(f'预测情感：{prediction} | 模型概率：{probs.max():.1%}')
            st.caption('模型概率未经独立校准，不代表判断一定正确。高亮是单独提取器选择的片段，不是分类器的因果解释。')
            highlighted=html.escape(text[:start])+'<mark>'+html.escape(span)+'</mark>'+html.escape(text[start+len(span):]) if start>=0 else html.escape(text)
            st.markdown('<div style="background:white;padding:24px;border-radius:14px;font-size:22px;white-space:pre-wrap">'+highlighted+'</div>',unsafe_allow_html=True)
            st.plotly_chart(px.bar(x=bundle['classifier'].classes_,y=probs,range_y=[0,1],labels={'x':'情感','y':'模型概率'}),width='stretch')
    st.subheader('批量分析')
    upload=st.file_uploader('上传含 text 列的 CSV（最多5,000条）',type='csv')
    if upload:
        try:
            batch=pd.read_csv(upload)
            if 'text' not in batch:st.error('CSV 必须有 text 列。')
            elif len(batch)>5000:st.error('请缩小到5,000条以内。')
            elif batch.text.isna().any():st.error('text 列有空值，请先补充或删除。')
            elif st.button('运行批量分析'):
                batch['prediction']=bundle['classifier'].predict(bundle['vectorizer'].transform(batch.text.astype(str)))
                batch['predicted_span']=[bundle['extractor'].predict_one(t,l) for t,l in zip(batch.text.astype(str),batch.prediction)]
                st.dataframe(batch,width='stretch')
                st.download_button('下载结果',batch.to_csv(index=False).encode('utf-8-sig'),'tweetlens_predictions.csv','text/csv')
        except Exception as e:st.error(f'无法读取数据：{e}')
with tabs[3]:
    only=st.checkbox('仅显示分类错误',True)
    f=df[df.prediction!=df.sentiment] if only else df
    label=st.selectbox('真实情感筛选',['全部']+LABELS)
    if label!='全部':f=f[f.sentiment==label]
    st.dataframe(f[['text','sentiment','prediction','confidence','gold_span','predicted_span','pipeline_jaccard']],width='stretch',hide_index=True)
    st.download_button('下载验证集预测',df.to_csv(index=False).encode('utf-8-sig'),'validation_predictions.csv','text/csv')
    st.caption('此页显示真实验证样本。否定、讽刺、混合情感等原因需要人工逐条检查，本项目不自动宣称已确认这些错误类型。')
with tabs[4]:
    st.markdown((ROOT/'METHODS.md').read_text())


with tabs[5]:
    st.subheader('RoBERTa frozen encoder + logistic regression')
    folder = ROOT / 'artifacts/roberta_frozen'
    if not (folder / 'metrics.json').exists():
        st.info('尚未完成本地训练与评估。运行 python roberta_frozen.py 生成真实结果。')
    else:
        result = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
        st.caption('固定预训练编码器，仅在原训练集拟合分类头。使用同一验证集。')
        a, b = st.columns(2)
        a.metric('Macro-F1', f"{result['metric']['macro_f1']:.3f}")
        b.metric('Accuracy', f"{result['metric']['accuracy']:.1%}")
        pair = pd.read_csv(folder / 'validation_predictions.csv')
        only_disagree = st.checkbox('只看与逻辑回归判断不同的推文')
        shown = pair[pair['Logistic regression'] != pair.roberta_prediction] if only_disagree else pair
        st.dataframe(shown[['text', 'sentiment', 'Logistic regression', 'roberta_prediction', 'roberta_confidence']], hide_index=True)
        text_roberta = st.text_area('用 RoBERTa 分析英文推文', key='roberta_text')
        if st.button('运行 RoBERTa'):
            if not text_roberta.strip():
                st.warning('请输入英文推文。')
            else:
                @st.cache_resource
                def roberta_resources():
                    from roberta_frozen import RobertaEncoder
                    return RobertaEncoder(), joblib.load(folder / 'head.joblib')
                with st.spinner('正在分析'):
                    encoder, classifier = roberta_resources()
                    probability = classifier.predict_proba(encoder.encode([text_roberta]))[0]
                st.success(str(classifier.classes_[probability.argmax()]))
                st.bar_chart(pd.DataFrame({'probability': probability}, index=classifier.classes_))
