import os
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdfplumber
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm

pdf_folder = "pdf"
output_csv = "salary_data.csv"
output_xlsx = "salary_data.xlsx"
output_chart = "salary_chart.png"

# Разбор PDF — CPU-нагрузка, поэтому по умолчанию процессы (быстрее потоков из-за GIL).
# Если нужны именно потоки, поставьте False.
USE_PROCESSES = True
MAX_WORKERS = min(32, os.cpu_count() or 4)

# ---------------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------------
# ФИО сотрудника, которого нужно выделить на графике (пунктир, поверх остальных).
# Можно писать полностью ("Иванов Иван Иванович") или частично ("Иванов Иван").
# Пустая строка — без выделения.
HIGHLIGHT_EMPLOYEE = ""

# True  — не учитывать сотрудников, у которых не было начислений за последние
#         INACTIVE_MONTHS месяцев (ни в таблице, ни в Excel, ни на графике).
# False — учитывать всех.
EXCLUDE_INACTIVE = True
INACTIVE_MONTHS = 3

# Имя файла: Расчетный листок__ФИО_MM_YYYY.pdf
# Шаблон намеренно мягкий: берём всё между префиксом "листок__" и суффиксом "_MM_YYYY.pdf".
filename_pattern = re.compile(
    r'^(.+?)[_\s]+(\d{2})[_\s.\-]+(\d{4})\.pdf$', re.IGNORECASE
)


def norm(s):
    """NFC-нормализация (имена файлов бывают с «й» из двух символов), nbsp -> пробел."""
    return unicodedata.normalize('NFC', s).replace('\xa0', ' ')


def clean_name(s):
    s = norm(s)
    s = re.sub(r'^.*?листок[\s_]*', '', s, flags=re.IGNORECASE)  # убрать префикс
    return ' '.join(s.replace('_', ' ').split())

# Резервный вариант: "за Январь 2020", "за января 2020" и т.п.
date_pattern = re.compile(r'за\s+([А-Яа-яЁё]+)\s+(\d{4})', re.IGNORECASE)

# Сопоставление по первым буквам, чтобы работали любые падежи
MONTH_PREFIXES = {
    'янв': '01', 'фев': '02', 'мар': '03', 'апр': '04',
    'мая': '05', 'май': '05', 'июн': '06', 'июл': '07',
    'авг': '08', 'сен': '09', 'окт': '10', 'ноя': '11', 'дек': '12',
}

# Сумма: цифры, пробел/nbsp как разделитель тысяч, запятая или точка в копейках.
# Только пробел и nbsp (не \s), чтобы не захватить следующую строку.
AMOUNT = r'(\d+(?:[ \xa0]\d{3})*(?:[.,]\d{1,2})?)'
accrued_pattern = re.compile(r'Начислено\s*:?[ \t\xa0]*' + AMOUNT, re.IGNORECASE)


def clean_amount(value):
    value = value.replace('\xa0', '').replace(' ', '').replace(',', '.')
    try:
        return float(value)
    except ValueError:
        return 0.0


def month_from_name(name):
    return MONTH_PREFIXES.get(name.lower()[:3])


def get_employee_date(filename, text):
    m = filename_pattern.match(norm(filename))
    if m:
        employee = clean_name(m.group(1))
        if employee:
            return employee, f'{m.group(3)}-{m.group(2)}'

    dm = date_pattern.search(text)
    month_year = ''
    if dm:
        mm = month_from_name(dm.group(1))
        if mm:
            month_year = f'{dm.group(2)}-{mm}'

    em = re.search(
        r'(?:ФИО|Сотрудник|Работник)\s*[:\-]\s*'
        r'([А-ЯЁ][А-ЯЁа-яё\-]+\s+[А-ЯЁа-яё\-]+\s+[А-ЯЁа-яё\-]+)',
        text, re.IGNORECASE,
    )
    if em:
        employee = clean_name(em.group(1))
    else:
        # Крайний случай: имя файла без даты и расширения
        base = re.sub(r'[_\s]*\d{2}[_\s.\-]+\d{4}$', '', os.path.splitext(norm(filename))[0])
        employee = clean_name(base)
    return employee, month_year


def process_pdf(filename):
    """Обрабатывает один PDF. Возвращает (строка_данных | None, предупреждение | None)."""
    filepath = os.path.join(pdf_folder, filename)
    try:
        with pdfplumber.open(filepath) as pdf:
            text = '\n'.join((page.extract_text() or '') for page in pdf.pages)
    except Exception as e:
        return None, f'Ошибка при обработке {filename}: {e}'

    employee, month_year = get_employee_date(filename, text)
    if not month_year:
        return None, f'[!] Не определён месяц: {filename}'

    m = accrued_pattern.search(text)
    warnings = []
    if not filename_pattern.match(norm(filename)):
        warnings.append(f'[!] Имя файла не по шаблону, ФИО взято из текста: {filename} -> {employee}')
    if not m:
        warnings.append(f'[!] Не найдено "Начислено": {filename}')
    warning = '\n'.join(warnings) or None
    row = {
        'month_year': month_year,
        'employee': employee,
        'salary': clean_amount(m.group(1)) if m else 0.0,
    }
    return row, warning


def main():
    pdf_files = sorted(f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf'))
    print(f'Найдено PDF-файлов: {len(pdf_files)}')
    if not pdf_files:
        raise SystemExit('PDF-файлы не найдены.')

    executor_cls = ProcessPoolExecutor if USE_PROCESSES else ThreadPoolExecutor
    print(f'Параллельная обработка: {executor_cls.__name__}, воркеров: {MAX_WORKERS}')

    rows = []
    with executor_cls(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_pdf, f) for f in pdf_files]
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc='Чтение PDF', unit='файл', ncols=100):
            row, warning = future.result()
            if warning:
                tqdm.write(warning)
            if row:
                rows.append(row)

    if not rows:
        raise SystemExit('Данные не удалось извлечь.')

    df = pd.DataFrame(rows)

    # Предупреждение, если у сотрудника несколько PDF за один месяц
    dups = df.duplicated(['month_year', 'employee'], keep=False)
    if dups.any():
        print('[!] Несколько PDF у одного сотрудника за один месяц (суммируются):')
        print(df[dups].sort_values(['employee', 'month_year']).to_string(index=False))

    # Строка = месяц, колонка = сотрудник, значение = Начислено (без средних)
    wide = (
        df.pivot_table(index='month_year', columns='employee',
                       values='salary', aggfunc='sum')
        .fillna(0)
        .sort_index()
    )
    wide = wide[sorted(wide.columns)]

    # Поиск выделяемого сотрудника (без учёта регистра, допускается часть ФИО)
    highlight = None
    if HIGHLIGHT_EMPLOYEE.strip():
        target = ' '.join(norm(HIGHLIGHT_EMPLOYEE).split()).casefold()
        exact = [c for c in wide.columns if c.casefold() == target]
        partial = [c for c in wide.columns if target in c.casefold()]
        if exact:
            highlight = exact[0]
        elif len(partial) == 1:
            highlight = partial[0]
        elif len(partial) > 1:
            print(f'[!] "{HIGHLIGHT_EMPLOYEE}" подходит нескольким сотрудникам, уточните ФИО: '
                  + '; '.join(partial[:10]))
        else:
            print(f'[!] Сотрудник "{HIGHLIGHT_EMPLOYEE}" не найден — выделение отключено.')

    # Исключение сотрудников без начислений за последние INACTIVE_MONTHS месяцев.
    # Отсчёт идёт от последнего месяца в данных, а не от сегодняшней даты.
    if EXCLUDE_INACTIVE:
        last = pd.Period(wide.index.max(), freq='M')
        recent = {str(last - i) for i in range(INACTIVE_MONTHS)}
        recent_rows = wide.loc[wide.index.isin(recent)]
        is_active = (recent_rows > 0).any(axis=0)
        keep = [c for c in wide.columns if is_active[c] or c == highlight]
        dropped = [c for c in wide.columns if c not in keep]
        if dropped:
            print(f'Исключено сотрудников без начислений за последние {INACTIVE_MONTHS} мес. '
                  f'(по {last}): {len(dropped)}')
            print('  ' + '; '.join(dropped[:30]) + (' ...' if len(dropped) > 30 else ''))
        wide = wide[keep]
        if dropped:
            # Месяцы, в которых остались одни нули (были только у исключённых), убираем
            wide = wide.loc[(wide > 0).any(axis=1)]
    print(f'Сотрудников: {wide.shape[1]}, месяцев: {len(wide)} -> {", ".join(wide.index)}')

    wide.reset_index().to_csv(output_csv, index=False, encoding='utf-8-sig')

    # Excel: максимум 16 384 колонок на лист, при необходимости делим на листы
    EXCEL_DATA_COLUMNS = 16384 - 1  # одна колонка под month_year
    employees = list(wide.columns)

    with pd.ExcelWriter(output_xlsx, engine='openpyxl') as writer:
        for n, start in enumerate(range(0, len(employees), EXCEL_DATA_COLUMNS), start=1):
            block = wide[employees[start:start + EXCEL_DATA_COLUMNS]].reset_index()
            block.to_excel(writer, index=False, sheet_name=f'Начислено_{n}')

    # График: только начисление по каждому месяцу
    fig, ax = plt.subplots(figsize=(18, 10))
    colors = plt.cm.hsv(np.linspace(0, 1, max(len(employees), 1), endpoint=False))

    for employee, color in zip(employees, colors):
        if employee == highlight:
            continue  # рисуем отдельно, последним
        ax.plot(wide.index, wide[employee], marker='o', linewidth=1.5,
                label=employee, color=color)

    if highlight:
        ax.plot(wide.index, wide[highlight], linestyle='--', marker='o',
                linewidth=3, markersize=7, color='black',
                label=f'{highlight} (выделен)', zorder=10)

    ax.set_title('Начисленная заработная плата по сотрудникам', fontsize=16)
    ax.set_xlabel('Месяц')
    ax.set_ylabel('Начислено, руб.')
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f'{x:,.0f}'.replace(',', ' '))
    )
    ax.grid(True, alpha=0.3)

    # Не больше ~45 строк в одной колонке легенды, чтобы она не вылезала по высоте
    legend_columns = max(1, -(-len(employees) // 45))
    ax.legend(title='Сотрудники', bbox_to_anchor=(1.02, 1), loc='upper left',
              ncol=legend_columns, fontsize=7)

    plt.xticks(rotation=45, ha='right')
    # tight_layout не используется: легенда вынесена за пределы графика, и он
    # не может её уместить. Нужные поля добавляет bbox_inches='tight' при сохранении.
    plt.savefig(output_chart, dpi=200, bbox_inches='tight')
    plt.close()

    print(f'Таблица сохранена: {output_csv}')
    print(f'Excel сохранен: {output_xlsx}')
    print(f'График сохранен: {output_chart}')


if __name__ == '__main__':
    main()
