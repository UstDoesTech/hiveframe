"""
Code Generation Module

Generate HiveFrame code from natural language descriptions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import re


@dataclass
class GeneratedCode:
    """Generated code with metadata"""
    code: str
    language: str  # 'python', 'sql', 'scala'
    description: str
    imports: List[str]
    confidence: float


@dataclass
class CodeTemplate:
    """Code template for common patterns"""
    name: str
    description: str
    template: str
    parameters: List[str]
    examples: List[str]


class CodeGenerator:
    """
    Generate code from natural language using pattern matching and templates.
    
    Like bees that follow genetic blueprints to build perfect hexagonal cells,
    this generator follows code templates to produce correct HiveFrame code.
    """
    
    def __init__(self):
        self.templates = self._initialize_templates()
        self.generation_history: List[Dict[str, Any]] = []
        
    def _initialize_templates(self) -> Dict[str, CodeTemplate]:
        """Initialize code templates for common operations"""
        return {
            "read_csv": CodeTemplate(
                name="read_csv",
                description="Read data from CSV file",
                template="""from hiveframe import HiveFrame

# Read CSV file
df = HiveFrame.read_csv("{file_path}")
print(f"Loaded {{len(df)}} rows")""",
                parameters=["file_path"],
                examples=["read data from file.csv", "load csv file"],
            ),
            
            "filter_data": CodeTemplate(
                name="filter_data",
                description="Filter DataFrame based on condition",
                template="""# Filter data
filtered_df = df.filter("{condition}")
print(f"Filtered to {{len(filtered_df)}} rows")""",
                parameters=["condition"],
                examples=["filter where age > 25", "keep only active users"],
            ),
            
            "group_aggregate": CodeTemplate(
                name="group_aggregate",
                description="Group by column and aggregate",
                template="""# Group by and aggregate
result = df.groupBy("{group_col}").agg({{
    "{agg_col}": "{agg_func}"
}})
print(result)""",
                parameters=["group_col", "agg_col", "agg_func"],
                examples=["count by category", "sum sales by region"],
            ),
            
            "join_tables": CodeTemplate(
                name="join_tables",
                description="Join two DataFrames",
                template="""# Join tables
result = df1.join(
    df2,
    on="{join_col}",
    how="{join_type}"
)
print(f"Joined result: {{len(result)}} rows")""",
                parameters=["join_col", "join_type"],
                examples=["join with other table", "merge on user_id"],
            ),
            
            "streaming": CodeTemplate(
                name="streaming",
                description="Process streaming data",
                template="""from hiveframe.streaming import HiveStream

# Create streaming pipeline
stream = HiveStream(source="{source}")
stream.window(size_sec={window_size}) \\
      .aggregate("{agg_func}") \\
      .sink("{sink}")

# Start streaming
stream.start()""",
                parameters=["source", "window_size", "agg_func", "sink"],
                examples=["stream from kafka", "process real-time data"],
            ),
        }
    
    def generate(self, description: str) -> GeneratedCode:
        """
        Generate code from natural language description.
        
        Args:
            description: Natural language description of desired code
            
        Returns:
            GeneratedCode object
        """
        desc_lower = description.lower()
        
        # Match to template
        best_template = None
        best_confidence = 0.0
        
        for template_name, template in self.templates.items():
            # Check if description matches template examples
            for example in template.examples:
                if example in desc_lower:
                    confidence = 0.8
                    if confidence > best_confidence:
                        best_template = template
                        best_confidence = confidence
                        break
        
        if not best_template:
            # Try keyword matching
            if any(word in desc_lower for word in ['read', 'load', 'csv', 'file']):
                best_template = self.templates['read_csv']
                best_confidence = 0.6
            elif any(word in desc_lower for word in ['filter', 'where', 'select']):
                best_template = self.templates['filter_data']
                best_confidence = 0.6
            elif any(word in desc_lower for word in ['group', 'aggregate', 'count', 'sum']):
                best_template = self.templates['group_aggregate']
                best_confidence = 0.6
            elif any(word in desc_lower for word in ['join', 'merge']):
                best_template = self.templates['join_tables']
                best_confidence = 0.6
            elif any(word in desc_lower for word in ['stream', 'real-time', 'kafka']):
                best_template = self.templates['streaming']
                best_confidence = 0.6
        
        if not best_template:
            # Generic fallback
            code = """# Generated HiveFrame code
from hiveframe import HiveFrame

# TODO: Implement specific logic
df = HiveFrame.read_csv("data.csv")
result = df.select("*")
print(result)"""
            return GeneratedCode(
                code=code,
                language='python',
                description="Generic HiveFrame template",
                imports=['from hiveframe import HiveFrame'],
                confidence=0.3,
            )
        
        # Extract parameters from description
        params = self._extract_parameters(description, best_template)
        
        # Fill template
        code = best_template.template
        for param, value in params.items():
            code = code.replace(f"{{{param}}}", str(value))
        
        # Extract imports
        imports = [line for line in code.split('\n') if line.strip().startswith('import') or line.strip().startswith('from')]
        
        result = GeneratedCode(
            code=code,
            language='python',
            description=best_template.description,
            imports=imports,
            confidence=best_confidence,
        )
        
        self.generation_history.append({
            "description": description,
            "template": best_template.name,
            "confidence": best_confidence,
        })
        
        return result
    
    def _extract_parameters(
        self,
        description: str,
        template: CodeTemplate,
    ) -> Dict[str, str]:
        """Extract parameter values from description"""
        params = {}
        desc_lower = description.lower()
        
        # Simple extraction heuristics
        if "file_path" in template.parameters:
            # Look for file names
            match = re.search(r'(\w+\.csv|\w+\.json)', description)
            params["file_path"] = match.group(1) if match else "data.csv"
        
        if "condition" in template.parameters:
            # Look for filter conditions
            if "where" in desc_lower:
                parts = desc_lower.split("where")
                if len(parts) > 1:
                    params["condition"] = parts[1].strip()
                else:
                    params["condition"] = "value > 0"
            else:
                params["condition"] = "value > 0"
        
        if "group_col" in template.parameters:
            # Look for group by column
            if "by" in desc_lower:
                parts = desc_lower.split("by")
                if len(parts) > 1:
                    col = parts[1].strip().split()[0]
                    params["group_col"] = col
                else:
                    params["group_col"] = "category"
            else:
                params["group_col"] = "category"
        
        if "agg_col" in template.parameters:
            params["agg_col"] = "value"
        
        if "agg_func" in template.parameters:
            if "count" in desc_lower:
                params["agg_func"] = "count"
            elif "sum" in desc_lower:
                params["agg_func"] = "sum"
            elif "avg" in desc_lower or "average" in desc_lower:
                params["agg_func"] = "avg"
            else:
                params["agg_func"] = "count"
        
        if "join_col" in template.parameters:
            # Look for join column
            if "on" in desc_lower:
                parts = desc_lower.split("on")
                if len(parts) > 1:
                    col = parts[1].strip().split()[0]
                    params["join_col"] = col
                else:
                    params["join_col"] = "id"
            else:
                params["join_col"] = "id"
        
        if "join_type" in template.parameters:
            if "left" in desc_lower:
                params["join_type"] = "left"
            elif "right" in desc_lower:
                params["join_type"] = "right"
            elif "outer" in desc_lower:
                params["join_type"] = "outer"
            else:
                params["join_type"] = "inner"
        
        if "source" in template.parameters:
            params["source"] = "kafka"
        
        if "window_size" in template.parameters:
            params["window_size"] = "60"
        
        if "sink" in template.parameters:
            params["sink"] = "console"
        
        return params


class HiveFrameCodeGen:
    """
    HiveFrame-specific code generation.
    
    Generates idiomatic HiveFrame code with bee-inspired patterns.
    """
    
    def __init__(self):
        self.generator = CodeGenerator()
        
    def generate_pipeline(self, steps: List[str]) -> GeneratedCode:
        """
        Generate a complete data pipeline from step descriptions.
        
        Args:
            steps: List of pipeline step descriptions
            
        Returns:
            GeneratedCode for the complete pipeline
        """
        code_parts = [
            "from hiveframe import HiveFrame",
            "from hiveframe.streaming import HiveStream",
            "",
            "# Data processing pipeline",
        ]
        
        for i, step in enumerate(steps):
            code = self.generator.generate(step)
            # Extract just the processing code (skip imports)
            processing_code = '\n'.join([
                line for line in code.code.split('\n')
                if not line.strip().startswith('import') and
                   not line.strip().startswith('from') and
                   line.strip() != ''
            ])
            code_parts.append(f"\n# Step {i+1}: {step}")
            code_parts.append(processing_code)
        
        full_code = '\n'.join(code_parts)
        
        return GeneratedCode(
            code=full_code,
            language='python',
            description="Multi-step data pipeline",
            imports=['from hiveframe import HiveFrame', 'from hiveframe.streaming import HiveStream'],
            confidence=0.7,
        )
    
    def generate_with_context(
        self,
        description: str,
        schema: Optional[Dict[str, List[str]]] = None,
    ) -> GeneratedCode:
        """
        Generate code with schema context for better accuracy.
        
        Args:
            description: Natural language description
            schema: Database schema for context
            
        Returns:
            GeneratedCode with context-aware generation
        """
        # Use schema to improve parameter extraction
        code = self.generator.generate(description)
        
        # If we have schema, validate and improve the generated code
        if schema:
            # Replace generic table/column names with actual ones from schema
            for table_name, columns in schema.items():
                if table_name.lower() in description.lower():
                    code.code = code.code.replace('data.csv', f'{table_name}.csv')
                    code.confidence = min(code.confidence + 0.1, 0.95)
        
        return code
