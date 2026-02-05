"""
AI-Powered Data Preparation

Automatic data cleaning and transformation suggestions using swarm intelligence.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class DataQualityIssue:
    """Detected data quality issue"""

    issue_type: str  # 'missing', 'outlier', 'format', 'duplicate', 'inconsistent'
    column: str
    severity: str  # 'low', 'medium', 'high'
    affected_rows: int
    description: str
    suggested_fix: str


@dataclass
class TransformationSuggestion:
    """Suggested data transformation"""

    transformation_type: str  # 'normalize', 'encode', 'aggregate', 'derive'
    columns: List[str]
    description: str
    code: str  # HiveFrame code to apply transformation
    confidence: float


class DataCleaner:
    """
    Automatic data cleaning using bee-inspired quality control.

    Like bees that remove debris and maintain hive cleanliness,
    this module identifies and fixes data quality issues.
    """

    def __init__(self):
        self.cleaning_rules: List[Dict[str, Any]] = []

    def analyze_quality(self, data: List[Dict[str, Any]]) -> List[DataQualityIssue]:
        """
        Analyze data quality and identify issues.

        Args:
            data: List of data records as dictionaries

        Returns:
            List of identified issues
        """
        if not data:
            return []

        issues = []
        columns = data[0].keys() if data else []

        for column in columns:
            # Check for missing values
            missing_count = sum(
                1 for row in data if row.get(column) is None or row.get(column) == ""
            )
            if missing_count > 0:
                severity = "high" if missing_count > len(data) * 0.3 else "medium"
                issues.append(
                    DataQualityIssue(
                        issue_type="missing",
                        column=column,
                        severity=severity,
                        affected_rows=missing_count,
                        description=f"{missing_count} missing values in {column}",
                        suggested_fix="Impute with median/mode or remove rows",
                    )
                )

            # Check for outliers (numeric columns)
            numeric_values = []
            for row in data:
                val = row.get(column)
                if isinstance(val, (int, float)) and val is not None:
                    numeric_values.append(val)

            if len(numeric_values) > 10:
                outliers = self._detect_outliers(numeric_values)
                if outliers:
                    issues.append(
                        DataQualityIssue(
                            issue_type="outlier",
                            column=column,
                            severity="medium",
                            affected_rows=len(outliers),
                            description=f"{len(outliers)} outliers detected in {column}",
                            suggested_fix="Apply winsorization or remove outliers",
                        )
                    )

            # Check for duplicates
            values = [row.get(column) for row in data]
            if len(values) != len(set(values)):
                duplicate_count = len(values) - len(set(values))
                issues.append(
                    DataQualityIssue(
                        issue_type="duplicate",
                        column=column,
                        severity="low",
                        affected_rows=duplicate_count,
                        description=f"{duplicate_count} duplicate values in {column}",
                        suggested_fix="Deduplicate or flag for review",
                    )
                )

        return issues

    def _detect_outliers(self, values: List[float]) -> List[float]:
        """Detect outliers using IQR method"""
        if len(values) < 4:
            return []

        sorted_values = sorted(values)
        q1 = sorted_values[len(sorted_values) // 4]
        q3 = sorted_values[3 * len(sorted_values) // 4]
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        return [v for v in values if v < lower_bound or v > upper_bound]

    def auto_clean(
        self,
        data: List[Dict[str, Any]],
        issues: List[DataQualityIssue],
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Automatically clean data based on identified issues.

        Args:
            data: Original data
            issues: Identified issues

        Returns:
            Tuple of (cleaned_data, actions_taken)
        """
        cleaned_data = [row.copy() for row in data]
        actions = []

        for issue in issues:
            if issue.issue_type == "missing" and issue.severity == "high":
                # Remove rows with missing values
                cleaned_data = [
                    row
                    for row in cleaned_data
                    if row.get(issue.column) is not None and row.get(issue.column) != ""
                ]
                actions.append(f"Removed {issue.affected_rows} rows with missing {issue.column}")

            elif issue.issue_type == "outlier":
                # Cap outliers (winsorization)
                values = [
                    row[issue.column]
                    for row in cleaned_data
                    if isinstance(row.get(issue.column), (int, float))
                ]
                if values:
                    p95 = sorted(values)[int(len(values) * 0.95)]
                    p5 = sorted(values)[int(len(values) * 0.05)]

                    for row in cleaned_data:
                        if isinstance(row.get(issue.column), (int, float)):
                            if row[issue.column] > p95:
                                row[issue.column] = p95
                            elif row[issue.column] < p5:
                                row[issue.column] = p5

                    actions.append(f"Capped outliers in {issue.column} at 5th and 95th percentiles")

        return cleaned_data, actions


class TransformationSuggester:
    """
    Suggest data transformations using swarm intelligence.

    Analyzes data patterns and suggests optimal transformations,
    similar to how bees collectively determine optimal foraging strategies.
    """

    def __init__(self):
        self.learned_patterns: Dict[str, List[str]] = {}

    def suggest_transformations(
        self,
        data: List[Dict[str, Any]],
        target_use_case: str = "general",
    ) -> List[TransformationSuggestion]:
        """
        Suggest data transformations.

        Args:
            data: Input data
            target_use_case: Intended use (e.g., 'ml', 'analytics', 'general')

        Returns:
            List of transformation suggestions
        """
        if not data:
            return []

        suggestions = []
        columns = data[0].keys() if data else []

        for column in columns:
            values = [row.get(column) for row in data if row.get(column) is not None]

            if not values:
                continue

            # Suggest encoding for categorical variables
            if all(isinstance(v, str) for v in values[:100]):
                unique_count = len(set(values))
                if unique_count < len(values) * 0.5:  # High cardinality
                    suggestions.append(
                        TransformationSuggestion(
                            transformation_type="encode",
                            columns=[column],
                            description=f"Encode categorical column {column}",
                            code=f"df.select('{column}').encode('onehot')",
                            confidence=0.8,
                        )
                    )

            # Suggest normalization for numeric variables
            if all(isinstance(v, (int, float)) for v in values[:100]):
                if target_use_case == "ml":
                    suggestions.append(
                        TransformationSuggestion(
                            transformation_type="normalize",
                            columns=[column],
                            description=f"Normalize {column} for ML",
                            code=f"df.select('{column}').normalize('minmax')",
                            confidence=0.7,
                        )
                    )

            # Suggest date parsing
            if all(isinstance(v, str) for v in values[:10]):
                if self._looks_like_date(values[0]):
                    suggestions.append(
                        TransformationSuggestion(
                            transformation_type="derive",
                            columns=[column],
                            description=f"Parse dates and extract features from {column}",
                            code=f"df.select('{column}').parse_date()"
                            ".extract_features(['year', 'month', 'day'])",
                            confidence=0.9,
                        )
                    )

        return suggestions

    def _looks_like_date(self, value: str) -> bool:
        """Check if string looks like a date"""
        date_patterns = [
            r"\d{4}-\d{2}-\d{2}",  # YYYY-MM-DD
            r"\d{2}/\d{2}/\d{4}",  # MM/DD/YYYY
            r"\d{2}-\d{2}-\d{4}",  # DD-MM-YYYY
        ]
        return any(re.match(pattern, value) for pattern in date_patterns)


class AIDataPrep:
    """
    AI-powered data preparation orchestrator.

    Combines cleaning and transformation suggestions to prepare
    data optimally for analysis or ML, inspired by how bee colonies
    process and store nectar into honey.
    """

    def __init__(self):
        self.cleaner = DataCleaner()
        self.suggester = TransformationSuggester()
        self.preparation_history: List[Dict[str, Any]] = []

    def prepare_data(
        self,
        data: List[Dict[str, Any]],
        target_use_case: str = "general",
        auto_clean: bool = True,
    ) -> Dict[str, Any]:
        """
        Prepare data with automatic cleaning and transformation suggestions.

        Args:
            data: Input data
            target_use_case: Intended use case
            auto_clean: Whether to automatically apply cleaning

        Returns:
            Dictionary with prepared data and metadata
        """
        # Analyze quality
        issues = self.cleaner.analyze_quality(data)

        # Clean data if requested
        cleaned_data = data
        cleaning_actions = []
        if auto_clean:
            cleaned_data, cleaning_actions = self.cleaner.auto_clean(data, issues)

        # Get transformation suggestions
        suggestions = self.suggester.suggest_transformations(cleaned_data, target_use_case)

        # Record preparation
        prep_record = {
            "original_rows": len(data),
            "cleaned_rows": len(cleaned_data),
            "issues_found": len(issues),
            "cleaning_actions": cleaning_actions,
            "suggestions": len(suggestions),
            "use_case": target_use_case,
        }
        self.preparation_history.append(prep_record)

        return {
            "data": cleaned_data,
            "issues": [
                {
                    "type": i.issue_type,
                    "column": i.column,
                    "severity": i.severity,
                    "description": i.description,
                    "suggested_fix": i.suggested_fix,
                }
                for i in issues
            ],
            "cleaning_actions": cleaning_actions,
            "transformation_suggestions": [
                {
                    "type": s.transformation_type,
                    "columns": s.columns,
                    "description": s.description,
                    "code": s.code,
                    "confidence": s.confidence,
                }
                for s in suggestions
            ],
            "metadata": prep_record,
        }
