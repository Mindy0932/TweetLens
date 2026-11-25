"""Rebuild the showcase from the original exported model and new measured results."""
from pathlib import Path
import json
import csv

ROOT=Path(__file__).resolve().parent
def build():
    payload=json.loads((ROOT/'artifacts/showcase_base.json').read_text(encoding='utf-8'))
    metrics=ROOT/'artifacts/roberta_frozen/metrics.json'
    if metrics.exists():
        payload['roberta']=json.loads(metrics.read_text(encoding='utf-8'))
        with (metrics.parent/'validation_predictions.csv').open(encoding='utf-8',newline='') as f:
            rows={r['textID']:r for r in csv.DictReader(f)}
        for row in payload['rows']:
            r=rows[row['textID']]
            row.update(roberta_prediction=r['roberta_prediction'],roberta_confidence=float(r['roberta_confidence']),roberta_probs=[float(r['roberta_'+label]) for label in ['negative','neutral','positive']])
    demo_file=ROOT/'artifacts/roberta_frozen/demo_predictions.json'
    if demo_file.exists():payload['robertaDemos']=json.loads(demo_file.read_text(encoding='utf-8'))
    template=(ROOT/'reports/template.html').read_text(encoding='utf-8')
    template=template.replace('<h3>05 / 来源与AI协助</h3>','<h3>05 / 作者与来源</h3><p><strong>Junyan Deng · 与 OpenAI Codex 协作开发。</strong></p>')
    template=template.replace('作品集应如实注明AI协助，并说明使用者实际完成的复现、修改与分析。','')
    template=template.replace('本版本由AI协助重新实现：','本项目在 AI 协助下实现：')
    template=template.replace('相同训练数据、相同验证集；超参数预先固定','同一训练划分与验证集。RoBERTa 使用外部预训练权重，固定编码器后拟合分类头。')
    template=template.replace('自动流程 Jaccard','原线性流程 Jaccard')
    template=template.replace('逻辑回归 · 行为真实标签，列为预测标签','行为真实标签，列为预测标签')
    template=template.replace('推理在浏览器本地完成','推理在本机完成')
    template=template.replace('真实逻辑回归推理，无需服务器或 API','两种模型均支持输入新推文进行分析。')
    template=template.replace('TF-IDF × LINEAR MODELS','TF-IDF × RoBERTa')
    extra=(ROOT/'reports/browser_inference.html').read_text(encoding='utf-8') + (ROOT/'reports/enhancements.html').read_text(encoding='utf-8') + (ROOT/'reports/chart_interactions.html').read_text(encoding='utf-8')
    locales=json.loads((ROOT/'reports/locales.json').read_text(encoding='utf-8'))
    locale_script=(ROOT/'reports/locale.html').read_text(encoding='utf-8')
    extra+=locale_script.replace('__LOCALE_DATA__',json.dumps(locales,ensure_ascii=False).replace('<','\u003c'))
    text=json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<','\u003c').replace('>','\u003e').replace('&','\u0026')
    page=template.replace('__PAYLOAD__',text).replace('</body>',extra+'</body>')
    (ROOT/'TweetLens_Showcase.html').write_text(page,encoding='utf-8')
    (ROOT/'index.html').write_text(page,encoding='utf-8')
    print('Built showcase; RoBERTa measured results:',bool(payload.get('roberta')))
if __name__=='__main__':build()
