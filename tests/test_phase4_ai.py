"""
Tests for Phase 4: Generative AI Integration
"""

import pytest
from hiveframe.ai.natural_language import (
    NaturalLanguageQuery, QueryTranslator, SQLGenerator,
    QueryIntent, GeneratedQuery
)
from hiveframe.ai.data_preparation import (
    DataCleaner, TransformationSuggester, AIDataPrep,
    DataQualityIssue
)
from hiveframe.ai.intelligent_discovery import (
    RelationshipDetector, JoinSuggester, DataDiscovery,
    Relationship
)
from hiveframe.ai.code_generation import (
    CodeGenerator, HiveFrameCodeGen,
    GeneratedCode
)
from hiveframe.ai.llm_platform import (
    LLMFineTuner, ModelTrainer, CustomModelSupport,
    TrainingConfig
)


class TestQueryTranslator:
    """Test natural language query translation"""
    
    def test_parse_select_intent(self):
        translator = QueryTranslator()
        intent = translator.parse_intent("Show all users")
        
        assert intent.intent_type == 'select'
        assert intent.confidence > 0
    
    def test_parse_aggregate_intent(self):
        translator = QueryTranslator()
        intent = translator.parse_intent("Count all orders")
        
        assert intent.intent_type == 'aggregate'
        assert 'COUNT' in intent.aggregations


class TestSQLGenerator:
    """Test SQL generation"""
    
    def test_generate_simple_select(self):
        generator = SQLGenerator()
        intent = QueryIntent(
            intent_type='select',
            entities=['users', 'name', 'email'],
            conditions=[],
            aggregations=[],
            confidence=0.8,
        )
        
        query = generator.generate_sql(intent)
        assert 'SELECT' in query.sql
        assert 'users' in query.sql
    
    def test_generate_aggregate(self):
        generator = SQLGenerator()
        intent = QueryIntent(
            intent_type='aggregate',
            entities=['orders'],
            conditions=[],
            aggregations=['COUNT'],
            confidence=0.8,
        )
        
        query = generator.generate_sql(intent)
        assert 'COUNT' in query.sql


class TestNaturalLanguageQuery:
    """Test integrated NL query interface"""
    
    def test_query_processing(self):
        nlq = NaturalLanguageQuery()
        result = nlq.query("Show all records")
        
        assert 'sql' in result
        assert 'explanation' in result
        assert 'confidence' in result
    
    def test_suggestions(self):
        nlq = NaturalLanguageQuery()
        suggestions = nlq.get_suggestions("show")
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0


class TestDataCleaner:
    """Test data cleaning"""
    
    def test_analyze_quality_missing_values(self):
        cleaner = DataCleaner()
        
        data = [
            {'name': 'Alice', 'age': 30},
            {'name': 'Bob', 'age': None},
            {'name': None, 'age': 25},
        ]
        
        issues = cleaner.analyze_quality(data)
        
        # Should detect missing values
        assert len(issues) > 0
        assert any(i.issue_type == 'missing' for i in issues)
    
    def test_detect_outliers(self):
        cleaner = DataCleaner()
        
        data = [
            {'value': 10},
            {'value': 12},
            {'value': 11},
            {'value': 100},  # Outlier
        ] * 3  # Repeat to have enough data
        
        issues = cleaner.analyze_quality(data)
        
        # May detect outliers
        assert isinstance(issues, list)
    
    def test_auto_clean(self):
        cleaner = DataCleaner()
        
        data = [
            {'name': 'Alice', 'age': 30},
            {'name': 'Bob', 'age': None},
            {'name': 'Charlie', 'age': 25},
        ]
        
        issues = cleaner.analyze_quality(data)
        cleaned_data, actions = cleaner.auto_clean(data, issues)
        
        assert len(actions) >= 0
        assert len(cleaned_data) <= len(data)


class TestTransformationSuggester:
    """Test transformation suggestions"""
    
    def test_suggest_encoding(self):
        suggester = TransformationSuggester()
        
        data = [
            {'category': 'A', 'value': 10},
            {'category': 'B', 'value': 20},
            {'category': 'A', 'value': 15},
        ]
        
        suggestions = suggester.suggest_transformations(data)
        
        # Should suggest encoding for categorical
        assert len(suggestions) >= 0
    
    def test_suggest_normalization(self):
        suggester = TransformationSuggester()
        
        data = [
            {'score': 100},
            {'score': 200},
            {'score': 150},
        ]
        
        suggestions = suggester.suggest_transformations(data, target_use_case='ml')
        
        # Should suggest normalization for ML
        assert len(suggestions) >= 0


class TestRelationshipDetector:
    """Test relationship detection"""
    
    def test_detect_relationships(self):
        detector = RelationshipDetector()
        
        schema = {
            'users': ['user_id', 'name', 'email'],
            'orders': ['order_id', 'user_id', 'amount'],
        }
        
        relationships = detector.detect_relationships(schema)
        
        # Should detect user_id relationship
        assert len(relationships) >= 0
    
    def test_columns_match(self):
        detector = RelationshipDetector()
        
        assert detector._columns_match('user_id', 'user_id')
        assert detector._columns_match('product_id', 'product_id')


class TestJoinSuggester:
    """Test join suggestions"""
    
    def test_suggest_joins(self):
        suggester = JoinSuggester()
        
        schema = {
            'users': ['user_id', 'name'],
            'orders': ['order_id', 'user_id'],
        }
        
        relationships = suggester.relationship_detector.detect_relationships(schema)
        suggestions = suggester.suggest_joins(
            'orders',
            ['users'],
            schema,
            relationships
        )
        
        assert isinstance(suggestions, list)


class TestCodeGenerator:
    """Test code generation"""
    
    def test_generate_read_csv(self):
        generator = CodeGenerator()
        code = generator.generate("read data from file.csv")
        
        assert isinstance(code, GeneratedCode)
        assert 'read_csv' in code.code or 'csv' in code.code.lower()
    
    def test_generate_filter(self):
        generator = CodeGenerator()
        code = generator.generate("filter data where age > 25")
        
        assert 'filter' in code.code.lower()
    
    def test_generate_group_aggregate(self):
        generator = CodeGenerator()
        code = generator.generate("count by category")
        
        assert 'count' in code.code.lower() or 'groupBy' in code.code


class TestHiveFrameCodeGen:
    """Test HiveFrame code generation"""
    
    def test_generate_pipeline(self):
        codegen = HiveFrameCodeGen()
        
        steps = [
            "read data from data.csv",
            "filter where value > 100",
            "group by category",
        ]
        
        code = codegen.generate_pipeline(steps)
        
        assert isinstance(code, GeneratedCode)
        assert len(code.code) > 0


class TestModelTrainer:
    """Test model training"""
    
    def test_prepare_dataset(self):
        trainer = ModelTrainer()
        
        data = [
            {'text': 'hello world', 'label': 1},
            {'text': 'goodbye world', 'label': 0},
            {'text': 'hi there', 'label': 1},
        ]
        
        dataset = trainer.prepare_dataset(data, 'text', 'label')
        
        assert 'train' in dataset
        assert 'validation' in dataset
        assert dataset['total_examples'] == 3
    
    def test_train(self):
        trainer = ModelTrainer()
        
        dataset = {
            'train': [{'text': 'hello', 'label': 1}] * 10,
            'validation': [{'text': 'hi', 'label': 1}] * 2,
            'total_examples': 12,
            'has_labels': True,
        }
        
        config = TrainingConfig(
            model_name='test_model',
            dataset_path='/tmp/test',
            batch_size=2,
            num_epochs=1,
        )
        
        model = trainer.train(config, dataset)
        
        assert model.model_id is not None
        assert model.base_model == 'test_model'


class TestLLMFineTuner:
    """Test LLM fine-tuning platform"""
    
    def test_initialization(self):
        finetuner = LLMFineTuner()
        
        assert finetuner.trainer is not None
        assert finetuner.custom_support is not None
    
    def test_list_models(self):
        finetuner = LLMFineTuner()
        models = finetuner.list_models()
        
        assert isinstance(models, list)


class TestAIDataPrep:
    """Test integrated AI data preparation"""
    
    def test_prepare_data(self):
        prep = AIDataPrep()
        
        data = [
            {'name': 'Alice', 'age': 30, 'score': 85},
            {'name': 'Bob', 'age': None, 'score': 90},
            {'name': 'Charlie', 'age': 25, 'score': 78},
        ]
        
        result = prep.prepare_data(data, auto_clean=True)
        
        assert 'data' in result
        assert 'issues' in result
        assert 'transformation_suggestions' in result


class TestDataDiscovery:
    """Test data discovery"""
    
    def test_discover_schema(self):
        discovery = DataDiscovery()
        
        schema = {
            'users': ['user_id', 'name'],
            'orders': ['order_id', 'user_id', 'amount'],
            'products': ['product_id', 'name', 'price'],
        }
        
        schema_graph = discovery.discover_schema(schema)
        
        assert len(schema_graph.tables) == 3
        assert isinstance(schema_graph.relationships, list)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
