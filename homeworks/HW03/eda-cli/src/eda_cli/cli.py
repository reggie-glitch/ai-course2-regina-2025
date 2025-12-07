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
        click.echo(f"Has missing values: {quality['has_missing']}")
        click.echo(f"Has duplicates: {quality['has_duplicates']}")
        
        # Покажем новые эвристики
        if quality['has_constant_columns']:
            click.echo(f"Constant columns: {', '.join(quality['constant_columns'])}")
        
        if quality['has_high_cardinality']:
            click.echo(f"High cardinality columns: {', '.join(quality['high_cardinality_cols'])}")
        
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
def report(csv_file, out_dir, title, max_hist_columns, top_k_categories,
           min_missing_share, high_cardinality_threshold, zero_threshold):
    """Generate EDA report."""
    try:
        df = load_data(csv_file)
        stats = compute_basic_stats(df)
        quality = compute_quality_flags(
            df, 
            high_cardinality_threshold=high_cardinality_threshold,
            zero_threshold=zero_threshold
        )
        
        # Создаем папку для отчета
        os.makedirs(out_dir, exist_ok=True)
        
        # Генерируем отчет
        report_path = Path(out_dir) / "report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# {title}\n\n")
            f.write(f"## Dataset Statistics\n")
            f.write(f"- Rows: {stats['rows']}\n")
            f.write(f"- Columns: {stats['columns']}\n\n")
            
            f.write(f"## Data Quality Assessment\n")
            f.write(f"- Overall quality check:\n")
            f.write(f"  - Has missing values: {quality['has_missing']}\n")
            f.write(f"  - Missing count: {quality['missing_count']}\n")
            f.write(f"  - Has duplicates: {quality['has_duplicates']}\n")
            f.write(f"  - Duplicate count: {quality['duplicate_count']}\n\n")
            
            # Новые эвристики из HW03
            f.write(f"## New Quality Heuristics (HW03)\n")
            
            if quality['has_constant_columns']:
                f.write(f"-  **Constant columns detected**: {', '.join(quality['constant_columns'])}\n")
            else:
                f.write(f"- ✓ No constant columns found\n")
            
            if quality['has_high_cardinality']:
                f.write(f"- **High cardinality columns** (> {quality.get('high_cardinality_threshold', 50)} unique values): {', '.join(quality['high_cardinality_cols'])}\n")
            else:
                f.write(f"- ✓ No high cardinality categorical columns\n")
            
            # Проверка дубликатов ID (если есть)
            if 'has_id_duplicates' in quality and quality['has_id_duplicates']:
                f.write(f"- **ID duplicates found**:\n")
                for col, count in quality.get('duplicate_id_info', {}).items():
                    f.write(f"  - {col}: {count} duplicates\n")
            
            # Проверка нулевых значений (если есть в core.py)
            if 'has_many_zeros' in quality and quality['has_many_zeros']:
                f.write(f"-  **Many zero values detected** (> {quality.get('zero_threshold', 0.3):.0%}):\n")
                for col, ratio in quality.get('zero_columns', []):
                    f.write(f"  - {col}: {ratio:.1%} zeros\n")
            
            f.write(f"\n## Report Generation Parameters\n")
            f.write(f"- Max histograms to generate: {max_hist_columns}\n")
            f.write(f"- Top categories to show: {top_k_categories}\n")
            f.write(f"- Missing values threshold: {min_missing_share:.0%}\n")
            f.write(f"- High cardinality threshold: {high_cardinality_threshold}\n")
            f.write(f"- Zero values threshold: {zero_threshold:.0%}\n")
            f.write(f"- Report generated with: eda-cli v0.1.0\n")
        
        click.echo(f"Report generated: {report_path}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

def main():
    cli()

if __name__ == '__main__':
    main()
