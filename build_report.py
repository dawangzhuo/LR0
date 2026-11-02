"""Rebuild the three-page Russian report from frozen CSV/JSON evidence."""
import csv
import json
import statistics
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'reports/source_data'

def read(name):
    return json.loads((DATA / name).read_text(encoding='utf-8-sig'))

def rows(name):
    with (DATA / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def main():
    # 在 Windows 使用 Arial；Linux 使用支持西里尔字母的 DejaVu。
    candidates = [Path('C:/Windows/Fonts/arial.ttf'), Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')]
    normal = next((p for p in candidates if p.is_file()), None)
    if normal is None:
        raise FileNotFoundError('A Cyrillic-capable Arial or DejaVuSans font is required.')
    bold = normal.with_name('arialbd.ttf' if normal.name == 'arial.ttf' else 'DejaVuSans-Bold.ttf')
    pdfmetrics.registerFont(TTFont('Body', str(normal)))
    pdfmetrics.registerFont(TTFont('BodyBold', str(bold)))
    pdfmetrics.registerFontFamily('Body', normal='Body', bold='BodyBold', italic='Body', boldItalic='BodyBold')
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='B', fontName='Body', fontSize=8.5, leading=12, spaceAfter=6))
    styles.add(ParagraphStyle(name='S', fontName='Body', fontSize=6.8, leading=9, textColor=colors.HexColor('#526273'), spaceAfter=7))
    styles.add(ParagraphStyle(name='H', fontName='BodyBold', fontSize=12, leading=16, spaceBefore=6, spaceAfter=8, textColor=colors.HexColor('#173854')))
    styles.add(ParagraphStyle(name='T', fontName='BodyBold', fontSize=19, leading=24, spaceAfter=12, textColor=colors.HexColor('#173854')))
    story, markdown = [], []
    def text(value, kind='B'):
        story.append(Paragraph(value, styles[kind]))
        markdown.append(value.replace('<b>','').replace('</b>',''))
    def table(headers, records, widths):
        content = [[Paragraph(escape(str(x)), styles['S']) for x in headers]]
        content += [[Paragraph(escape(str(x)), styles['B']) for x in record] for record in records]
        t = Table(content, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eaf1f6')),
                              ('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),3),
                              ('TOPPADDING',(0,0),(-1,-1),3),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#b2c4d1')),
                              ('LINEBELOW',(0,-1),(-1,-1),.4,colors.HexColor('#b2c4d1'))]))
        story.append(t);story.append(Spacer(1,7))
        markdown.append('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in records))
    def page():
        story.append(PageBreak());markdown.append('\n---\n')
    base, final, search = read('base_statistics.json'), read('final_statistics.json'), read('search_analysis.json')
    br, pairs, pred = rows('base_seeds_summary.csv'), rows('paired_comparison.csv'), rows('paired_predictions.csv')
    pct=lambda v:f'{float(v)*100:.4f}%'
    pp=lambda v:f'{float(v):+.4f}'
    text('ЛР0 · Воспроизводимый эксперимент', 'T')
    text('Подбор гиперпараметров MLP: превышает ли улучшение случайный разброс?', 'H')
    text('Проверяемая гипотеза: настройка должна давать выигрыш, превышающий разброс базовой модели по seed. Исходный прогноз хранится отдельно в prediction.md; этот итоговый текст составлен после экспериментов и не заменяет исходный прогноз. Численный прогноз здесь не приписывается исходному документу, копия которого не предоставлена для подготовки отчёта.')
    text('1. Метод и воспроизводимость', 'H')
    text('Данные load_digits: 1797 изображений 8×8, 10 классов; признаки разделены на 16. Стратифицированное разбиение: train 1077, val 360, test 360; отдельный split_seed=0. Обучение: MLP с ReLU и dropout, Adam, cross-entropy, 30 эпох, CPU, один поток. train.py сохраняет конфигурацию, metadata.json, историю, предсказания и метрики; Optuna вызывает ту же процедуру обучения. Итоговые метрики пересчитываются из predictions.csv. Контроль цепочки: конфигурация + Git + SHA-256 → train.py → runs/ и MLflow → скрипты анализа → таблицы и графики.')
    text('Разбиение SHA-256: '+final['source_conditions']['split_sha256']+'. Исходный коммит финальной серии: '+final['source_conditions']['commit']+'. Python '+final['source_conditions']['python']+'; версии библиотек сохранены в metadata.json и final_statistics.json.', 'S')
    text('Ограничение протокола: до формальной серии были разведочные запуски, включая просмотр test при другом разбиении. Они исключены из таблиц; полную слепоту ко всему набору данных утверждать нельзя. В формальном поиске test не использовался, best.yaml заморожен до десяти финальных запусков; после их оценки конфигурация не перенастраивалась.')
    text('2. Эксперимент 1: разброс базовой модели', 'H')
    table(['Seed','Верно / N','Val accuracy','Время, с'],[[r['seed'],r['val_correct']+' / '+r['val_samples'],pct(r['val_acc']),f"{float(r['training_seconds']):.6f}"] for r in br],[45,110,170,174])
    text('Источник: серия base_seeds, seeds 1–5; reports/source_data/base_seeds_summary.csv. Код: summarize_base.py. Каждая строка - 360 изображений val; run_id и коммиты перечислены в CSV.', 'S')
    text(f"Минимум / медиана / максимум: {pct(base['accuracy_min'])} / {pct(base['accuracy_median'])} / {pct(base['accuracy_max'])}. Размах: {base['accuracy_range_pp']:.4f} п.п. Два формальных запуска seed=0 дали одинаковые метрики (без времени) и предсказания, val=95.0000%. Для seed=2 относительно seed=1 исправлен {base['paired_comparison']['corrected']} ответ, испорчено {base['paired_comparison']['spoiled']} из 360: Δ={base['paired_comparison']['accuracy_delta']*100:+.4f} п.п.")
    times = [float(r['training_seconds']) for r in br]
    text(f'Время базовой серии: минимум / медиана / максимум = {min(times):.6f} / {statistics.median(times):.6f} / {max(times):.6f} с.')
    text('Источники повторяемости: runs/20261024_133833_264af668a0fd и runs/20261024_133936_81ffb4c2f6fd; попарный анализ: seed1_vs_seed2.csv, summarize_base.py. Время - сумма интервалов обучения и валидации по эпохам, без настройки, записи файлов и MLflow.', 'S')
    page()
    text('3. Эксперимент 2: поиск Optuna', 'T')
    text('Последовательный TPE (sampler seed=42), MedianPruner (startup=5, warmup=5), seed обучения=0. Study lr0_search сохранена в optuna.db: первые 10 trial, затем продолжение до 40. Возобновление сохраняет историю trial, но повторная инициализация sampler не гарантирует последовательность предложений непрерывного поиска.')
    text('Пространство: lr 10⁻⁴–10⁻¹ (log); n_layers 1–3; hidden 32–512 (int, log); dropout 0–0.5; weight_decay 10⁻⁶–10⁻² (log); batch_size ∈ {32, 64, 128}. Цель - accuracy на val после последней эпохи, не максимум по эпохам.')
    text(f"Всего {search['total_trials']} trial: COMPLETE={search['state_counts']['COMPLETE']}, PRUNED={search['state_counts']['PRUNED']}, FAIL={search['state_counts']['FAIL']}. Выполнено {search['executed_epochs']} / {search['full_epoch_budget']} эпох; не выполнено {search['unexecuted_epochs']} ({search['unexecuted_epoch_fraction']*100:.2f}%). Измерено {search['measured_training_seconds']:.6f} с обучения и валидации. Оценка экономии: {search['estimated_pruning_saved_seconds']:.6f} с; для каждого прерванного trial оставшиеся эпохи умножены на его среднее время эпохи. Это не прямой замер поиска без прунинга.")
    for name, caption in [('search_progress.png','Рис. 1. Точность на val по trial (N=360); PRUNED показывает промежуточную, а не финальную оценку. Линия - лучший завершённый trial.'),('parameter_importance.png','Рис. 2. Относительная важность параметров по 13 COMPLETE trial (fANOVA); оценка ограничена малой выборкой и не является причинным эффектом.')]:
        image = Image(str(ROOT/'reports/figures'/name), width=365, height=205.3 if name.startswith('search') else 234.6)
        story.append(image)
        text(caption+' Источник: lr0_search, optuna.db; код: analyze_search.py. Сводка: search_analysis.json.', 'S')
        markdown.append(f'![{caption}](figures/{name})')
    params=search['best_params']
    text(f"Лучший trial №{search['best_trial']} (нумерация с нуля): val={pct(search['best_val_accuracy'])}. Выбраны n_layers={params['n_layers']}, hidden={params['hidden']}, dropout={params['dropout']:.6f}, lr={params['lr']:.8f}, weight_decay={params['weight_decay']:.8g}, batch_size={params['batch_size']}. Точные значения: configs/best.yaml.")
    page()
    text('4. Эксперимент 3: парное сравнение', 'T')
    text('После замораживания конфигурации выполнены final_base и final_best с новыми seeds 101–105: десять запусков при общих условиях и коммите. Веса обучаются только на train=1077; val не присоединяется. Оценка test включена только в финальной формальной серии. Δ = best − base; п.п. - процентные пункты.')
    table(['Seed','Base test','Best test','Δ val, п.п.','Δ test, п.п.'],[[r['seed'],pct(r['base_test_acc']),pct(r['best_test_acc']),pp(r['delta_val_pp']),pp(r['delta_test_pp'])] for r in pairs],[40,113,113,113,120])
    text('Источник: final_base / final_best, seeds 101–105; paired_comparison.csv и final_runs.csv. Каждая accuracy имеет знаменатель 360; код: summarize_final.py. Полные run_id сохранены в final_runs.csv.', 'S')
    text(f"Δ val: минимум / медиана / максимум = {pp(final['delta_val_pp']['min'])} / {pp(final['delta_val_pp']['median'])} / {pp(final['delta_val_pp']['max'])} п.п.; Δ test = {pp(final['delta_test_pp']['min'])} / {pp(final['delta_test_pp']['median'])} / {pp(final['delta_test_pp']['max'])} п.п. Все пять test-разностей положительны и превышают размах базовой серии {final['baseline_accuracy_range_pp']:.4f} п.п. По критерию задания улучшение устойчиво в исследованных условиях.")
    text(f"Лучшая поисковая val accuracy: {pct(final['search_best_val_acc'])}; средняя val accuracy выбранной конфигурации на новых seed: {pct(final['best_new_seeds_mean_val_acc'])}; средняя test accuracy: {pct(final['best_new_seeds_mean_test_acc'])}. Снижение совместимо со смещением отбора и различиями seed / выборок; эти причины данным опытом отдельно не измерены.")
    text(f"На test исправлено {final['test_corrected_total']}, испорчено {final['test_spoiled_total']} ответов суммарно по пяти seed. Чистый выигрыш {final['test_corrected_total']-final['test_spoiled_total']} / {final['test_paired_observations']} = {(final['test_corrected_total']-final['test_spoiled_total'])/final['test_paired_observations']*100:.4f} п.п. Это {final['test_paired_observations']} парных наблюдений на тех же {final['unique_test_images']} уникальных изображениях, а не 1800 независимых изображений.")
    examples=[]
    for outcome in ['corrected','spoiled']:
        r=next(r for r in pred if r['split']=='test' and r['seed']=='101' and r['outcome']==outcome)
        examples.append([r['seed'],r['index'],r['y_true'],r['base_prediction'],r['best_prediction'],'Исправлено' if outcome=='corrected' else 'Испорчено'])
    table(['Seed','Индекс','Класс','Base','Best','Исход'],examples,[45,65,65,65,65,194])
    text('Источник: paired_predictions.csv, split=test, seed=101; по одной паре предсказаний в строке. Код: summarize_final.py; выбор примеров: build_report.py. Эти случаи показывают изменение ошибок, но без анализа изображения не объясняют внутреннюю причину решения сети.', 'S')
    text('5. Вывод, ограничения и кодогенерация', 'H')
    text('Подбор улучшил итоговую точность больше наблюдаемого шума от seed. Вывод относится к этому фиксированному разбиению digits, MLP и пяти новым seed; это не тест статистической значимости и не гарантия на других данных. Важность параметров и экономия времени интерпретируются с оговорками выше.')
    text('С помощью ChatGPT подготовлены и исправлены код каркаса, поиска, анализа, документация и этот отчёт. Пользователь выполнил команды локально. Проверены компиляция, границы lr, отказ при dirty Git, восстановление метрик из сохранённых предсказаний, повторяемость seed=0, различия seed=1, полнота серий, единые условия, 40 сохранённых trial и соответствие таблиц CSV/JSON. Архивирование и упаковка выполняются отдельно и не являются новыми экспериментами.')
    text('Источники: задание «Лабораторная работа 0» (§3–5), «Общие правила» (§6–8); исходные CSV/JSON в reports/source_data, графики в reports/figures. Рендер и все численные вставки: build_report.py. Исходный прогноз: prediction.md (проверить соответствие формулировок перед сдачей).', 'S')
    def footer(canvas, doc):
        canvas.setFont('Body',7)
        canvas.setFillColor(colors.HexColor('#526273'))
        canvas.drawString(48,25,'ЛР0  |  Сохранённые результаты, без повторного обучения')
        canvas.drawRightString(547,25,str(doc.page))
    SimpleDocTemplate(str(ROOT/'reports/report.pdf'), pagesize=(595,842), leftMargin=48,
                      rightMargin=48, topMargin=36, bottomMargin=40).build(story,onFirstPage=footer,onLaterPages=footer)
    (ROOT/'reports/report.md').write_text('\n\n'.join(markdown)+'\n',encoding='utf-8')
    print('Report generated: reports/report.pdf and reports/report.md')

if __name__=='__main__':
    main()
