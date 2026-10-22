# DataMind AI - Agent Tools

from src.tools.data_loader import (
    load_dataset,
    get_dataset,
    get_dataset_metadata,
    describe_dataset,
    list_datasets,
    update_dataset,
    delete_dataset,
)
from src.tools.sql_engine import execute_sql, get_table_schema
from src.tools.data_cleaner import (
    auto_clean_dataset,
    handle_missing_values,
    handle_duplicates,
    infer_and_convert_types,
    detect_and_handle_outliers,
    clean_column_names,
)
from src.tools.data_profiler import profile_dataset, compute_group_aggregations
from src.tools.statistical import (
    run_ttest,
    run_anova,
    run_chi_square,
    run_correlation_test,
    run_normality_test,
)
from src.tools.ml_tools import (
    train_and_evaluate_model,
    train_clustering,
    auto_detect_problem_type,
)
from src.tools.chart_builder import (
    create_bar_chart,
    create_line_chart,
    create_scatter_plot,
    create_histogram,
    create_box_plot,
    create_correlation_heatmap,
    create_pie_chart,
    recommend_charts_for_dataset,
)
from src.tools.report_generator import generate_html_report

__all__ = [
    "load_dataset",
    "get_dataset",
    "describe_dataset",
    "list_datasets",
    "update_dataset",
    "delete_dataset",
    "execute_sql",
    "get_table_schema",
    "auto_clean_dataset",
    "handle_missing_values",
    "handle_duplicates",
    "infer_and_convert_types",
    "detect_and_handle_outliers",
    "clean_column_names",
    "profile_dataset",
    "compute_group_aggregations",
    "run_ttest",
    "run_anova",
    "run_chi_square",
    "run_correlation_test",
    "run_normality_test",
    "train_and_evaluate_model",
    "train_clustering",
    "auto_detect_problem_type",
    "create_bar_chart",
    "create_line_chart",
    "create_scatter_plot",
    "create_histogram",
    "create_box_plot",
    "create_correlation_heatmap",
    "create_pie_chart",
    "recommend_charts_for_dataset",
    "generate_html_report",
]
