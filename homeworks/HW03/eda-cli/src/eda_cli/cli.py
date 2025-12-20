import click
import pandas as pd
from pathlib import Path
import sys
import os

from eda_cli.core import load_data, compute_basic_stats, compute_quality_flags

@click.group()
def cli():
    """Mini EDA CLI tool."""
    pass

@cli.command()
@click.argument('csv_file', type=click.Path(exists=True))
def overview(csv_file):
    """Show dataset overview."""
    try:
        df = load_data(csv_file)
        stats = compute_basic_stats(df)
        quality = compute_quality_flags(df)
        
        click.echo("Dataset Overview")
        click.echo(f"Rows: {stats['rows']}")
        click.echo(f"Columns: {stats['columns']}")
        click.echo(f"Has missing values: {quality.get('has_missing', False)}")
        click.echo(f"Has duplicates: {quality.get('has_duplicates', False)}")
        
        # Покажем новые эвристики
        if quality.get('has_constant_columns', False):
            click.echo(f"Constant columns: {', '.join(quality.get('constant_columns', []))}")
        
        if quality.get('has_high_cardinality', False):
            click.echo(f"High cardinality columns: {', '.join(quality.get('high_cardinality_cols', []))}")
        
        
        if quality.get('has_outliers', False):
            click.echo(f"Outliers detected in columns")
        
        if quality.get('has_imbalanced_categories', False):
            click.echo(f"Imbalanced categories detected")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('csv_file', type=click.Path(exists=True))
@click.option('--out-dir', default='reports', help='Output directory')
@click.option('--title', default='EDA Report', help='Report title')
@click.option('--max-hist-columns', default=5, type=int, 
              help='Max histograms to generate')
@click.option('--top-k-categories', default=10, type=int,
              help='Top categories to show')
@click.option('--min-missing-share', default=0.1, type=float,
              help='Missing share threshold')
@click.option('--high-cardinality-threshold', default=50, type=int,
              help='High cardinality threshold')
@click.option('--zero-threshold', default=0.3, type=float,
              help='Zero values threshold')

@click.option('--outlier-threshold', default=1.5, type=float,
              help='IQR multiplier for outlier detection (default: 1.5)')
@click.option('--imbalance-threshold', default=0.9, type=float,
              help='Threshold for imbalanced categories (default: 0.9)')
def report(csv_file, out_dir, title, max_hist_columns, top_k_categories,
           min_missing_share, high_cardinality_threshold, zero_threshold,
           outlier_threshold, imbalance_threshold):  # НОВЫЕ ПАРАМЕТРЫ
    """Generate EDA report with extended heuristics."""
    try:
        df = load_data(csv_file)
        stats = compute_basic_stats(df)
        
        # Вызываем compute_quality_flags со ВСЕМИ параметрами
        quality = compute_quality_flags(
            df, 
            high_cardinality_threshold=high_cardinality_threshold,
            zero_threshold=zero_threshold,
            min_missing_share=min_missing_share,
            outlier_threshold=outlier_threshold,      # НОВЫЙ
            imbalance_threshold=imbalance_threshold   # НОВЫЙ
        )
        
        
        os.makedirs(out_dir, exist_ok=True)
        
        
        report_path = Path(out_dir) / "report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# {title}\n\n")
            f.write(f"## Dataset Statistics\n")
            f.write(f"- Rows: {stats['rows']}\n")
            f.write(f"- Columns: {stats['columns']}\n\n")
            
            f.write(f"## Data Quality Assessment\n")
            f.write(f"- Overall quality check:\n")
            f.write(f"  - Has missing values: {quality.get('has_missing', False)}\n")
            f.write(f"  - Missing count: {quality.get('missing_count', 0)}\n")
            f.write(f"  - Has duplicates: {quality.get('has_duplicates', False)}\n")
            f.write(f"  - Duplicate count: {quality.get('duplicate_count', 0)}\n\n")
            
            f.write(f"## Quality Heuristics (HW03)\n")
            
           
            if quality.get('has_constant_columns', False):
                f.write(f"- **Constant columns detected**: {', '.join(quality.get('constant_columns', []))}\n")
            else:
                f.write(f"- ✓ No constant columns found\n")
            
            if quality.get('has_high_cardinality', False):
                f.write(f"- **High cardinality columns** (> {high_cardinality_threshold} unique values): {', '.join(quality.get('high_cardinality_cols', []))}\n")
            else:
                f.write(f"- ✓ No high cardinality categorical columns\n")
            
           
            if quality.get('has_id_duplicates', False):
                f.write(f"- **ID duplicates found**:\n")
                for col, info in quality.get('duplicate_id_info', {}).items():
                    f.write(f"  - {col}: {info.get('duplicate_count', 0)} duplicates\n")
            
           
            if quality.get('has_many_zeros', False):
                f.write(f"- **Many zero values detected** (> {zero_threshold:.0%}):\n")
                for col_info in quality.get('zero_columns', []):
                    f.write(f"  - {col_info.get('column', 'unknown')}: {col_info.get('zero_ratio', 0):.1%} zeros\n")
            
            
            f.write(f"\n## Extended Heuristics (HW03 Extension)\n")
            
            # Outliers
            if quality.get('has_outliers', False):
                f.write(f"- **Outliers detected** (IQR threshold: {outlier_threshold}):\n")
                for col_info in quality.get('outlier_columns', []):
                    f.write(f"  - {col_info.get('column', 'unknown')}: ")
                    f.write(f"{col_info.get('outliers_count', 0)} outliers ")
                    f.write(f"({col_info.get('outliers_ratio', 0):.1%})\n")
            else:
                f.write(f"- ✓ No significant outliers detected\n")
            
            # Imbalanced categories
            if quality.get('has_imbalanced_categories', False):
                f.write(f"- **Imbalanced categories detected** (threshold: {imbalance_threshold:.0%}):\n")
                for col_info in quality.get('imbalanced_columns', []):
                    f.write(f"  - {col_info.get('column', 'unknown')}: ")
                    f.write(f"'{col_info.get('dominant_category', 'unknown')}' is ")
                    f.write(f"{col_info.get('dominant_ratio', 0):.1%} of data\n")
            else:
                f.write(f"- ✓ Categories are reasonably balanced\n")
            
            f.write(f"\n## Report Generation Parameters\n")
            f.write(f"- Max histograms to generate: {max_hist_columns}\n")
            f.write(f"- Top categories to show: {top_k_categories}\n")
            f.write(f"- Missing values threshold: {min_missing_share:.0%}\n")
            f.write(f"- High cardinality threshold: {high_cardinality_threshold}\n")
            f.write(f"- Zero values threshold: {zero_threshold:.0%}\n")
            # Новые параметры
            f.write(f"- Outlier detection threshold (IQR multiplier): {outlier_threshold}\n")
            f.write(f"- Imbalance threshold: {imbalance_threshold:.0%}\n")
            f.write(f"- Report generated with: eda-cli v0.2.0\n")
        
        click.echo(f"Report generated: {report_path}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

def main():
    cli()

if __name__ == '__main__':
    main()
