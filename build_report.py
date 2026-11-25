"""Refresh the report’s embedded charts and repository hyperlink."""
import argparse
from pathlib import Path

from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

REPOSITORY = 'https://github.com/wyhg4768-boop/cell2cell-churn-prediction'


def replace_link(doc, url):
    paragraph, = [p for p in doc.paragraphs if p.text.startswith('GitHub 项目链接')]
    for item in list(paragraph._p):
        if item.tag != qn('w:pPr'):
            paragraph._p.remove(item)
    paragraph.add_run('GitHub 项目链接  ')
    hyperlink = OxmlElement('w:hyperlink')
    hyperlink.set(qn('r:id'),doc.part.relate_to(url,RELATIONSHIP_TYPE.HYPERLINK,is_external=True))
    run = OxmlElement('w:r')
    properties = OxmlElement('w:rPr')
    underline = OxmlElement('w:u')
    underline.set(qn('w:val'),'single')
    color = OxmlElement('w:color')
    color.set(qn('w:val'),'0563C1')
    properties.extend((color, underline))
    label = OxmlElement('w:t')
    label.text = url
    run.extend((properties,label))
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def run(source, output, eda, grid, formula, repository):
    doc = Document(source)
    if any(value is not None for value in (eda, grid, formula)):
        for shape in doc.inline_shapes:
            name = shape._inline.docPr.get('descr', '')
            if eda is not None and name.endswith('的分组流失率'):
                image = eda/'eda_groups.png'
            elif eda is not None and name.startswith('留存组与流失组'):
                image = eda/'eda_distributions.png'
            elif grid is not None and name.startswith('XGBoost 的 ROC'):
                image = grid/'report_curves.png'
            elif formula is not None and name.startswith('客户特征 x_i'):
                image = formula
            else:
                continue
            rid = shape._inline.graphic.graphicData.pic.blipFill.blip.embed
            doc.part.related_parts[rid]._blob = image.read_bytes()
    replace_link(doc, repository)
    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('report/基于客户信息的电信客户流失预测报告.docx'))
    parser.add_argument('--output', type=Path, default=Path('runs/report/电信客户流失预测报告.docx'))
    parser.add_argument('--eda', type=Path)
    parser.add_argument('--grid', type=Path)
    parser.add_argument('--formula', type=Path)
    parser.add_argument('--repository', default=REPOSITORY)
    args = parser.parse_args()
    run(args.source,args.output,args.eda,args.grid,args.formula,args.repository)
