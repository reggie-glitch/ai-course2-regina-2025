# Тест новых эвристик

## Dataset Statistics
- Rows: 10
- Columns: ['id', 'name', 'age', 'salary', 'department', 'country', 'score', 'is_active']

## Data Quality Assessment
- Overall quality check:
  - Has missing values: False
  - Missing count: 0
  - Has duplicates: False
  - Duplicate count: 0

## Quality Heuristics (HW03)
- ✓ No constant columns found
- ✓ No high cardinality categorical columns

## Extended Heuristics (HW03 Extension)
- ✓ No significant outliers detected
- ✓ Categories are reasonably balanced

## Report Generation Parameters
- Max histograms to generate: 5
- Top categories to show: 10
- Missing values threshold: 10%
- High cardinality threshold: 50
- Zero values threshold: 30%
- Outlier detection threshold (IQR multiplier): 2.0
- Imbalance threshold: 85%
- Report generated with: eda-cli v0.2.0
