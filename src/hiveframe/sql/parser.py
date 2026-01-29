"""
SQL Parser
==========

Tokenizer and parser for SwarmQL SQL syntax.
Converts SQL strings into structured AST representations.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union
import re


class TokenType(Enum):
    """SQL token types."""
    # Keywords
    SELECT = auto()
    FROM = auto()
    WHERE = auto()
    GROUP = auto()
    BY = auto()
    HAVING = auto()
    ORDER = auto()
    LIMIT = auto()
    OFFSET = auto()
    JOIN = auto()
    INNER = auto()
    LEFT = auto()
    RIGHT = auto()
    OUTER = auto()
    FULL = auto()
    CROSS = auto()
    ON = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    IN = auto()
    BETWEEN = auto()
    LIKE = auto()
    IS = auto()
    NULL = auto()
    AS = auto()
    ASC = auto()
    DESC = auto()
    DISTINCT = auto()
    ALL = auto()
    UNION = auto()
    EXCEPT = auto()
    INTERSECT = auto()
    CASE = auto()
    WHEN = auto()
    THEN = auto()
    ELSE = auto()
    END = auto()
    CAST = auto()
    TRUE = auto()
    FALSE = auto()
    
    # Identifiers and literals
    IDENTIFIER = auto()
    NUMBER = auto()
    STRING = auto()
    
    # Operators
    EQUALS = auto()
    NOT_EQUALS = auto()
    LESS_THAN = auto()
    LESS_EQUALS = auto()
    GREATER_THAN = auto()
    GREATER_EQUALS = auto()
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    
    # Punctuation
    COMMA = auto()
    DOT = auto()
    LPAREN = auto()
    RPAREN = auto()
    SEMICOLON = auto()
    
    # Special
    EOF = auto()
    

@dataclass
class Token:
    """SQL token with position information."""
    type: TokenType
    value: str
    line: int
    column: int


class SQLTokenizer:
    """
    SQL Tokenizer
    -------------
    Converts SQL strings into token streams.
    """
    
    # Keyword mapping
    KEYWORDS = {
        'SELECT': TokenType.SELECT,
        'FROM': TokenType.FROM,
        'WHERE': TokenType.WHERE,
        'GROUP': TokenType.GROUP,
        'BY': TokenType.BY,
        'HAVING': TokenType.HAVING,
        'ORDER': TokenType.ORDER,
        'LIMIT': TokenType.LIMIT,
        'OFFSET': TokenType.OFFSET,
        'JOIN': TokenType.JOIN,
        'INNER': TokenType.INNER,
        'LEFT': TokenType.LEFT,
        'RIGHT': TokenType.RIGHT,
        'OUTER': TokenType.OUTER,
        'FULL': TokenType.FULL,
        'CROSS': TokenType.CROSS,
        'ON': TokenType.ON,
        'AND': TokenType.AND,
        'OR': TokenType.OR,
        'NOT': TokenType.NOT,
        'IN': TokenType.IN,
        'BETWEEN': TokenType.BETWEEN,
        'LIKE': TokenType.LIKE,
        'IS': TokenType.IS,
        'NULL': TokenType.NULL,
        'AS': TokenType.AS,
        'ASC': TokenType.ASC,
        'DESC': TokenType.DESC,
        'DISTINCT': TokenType.DISTINCT,
        'ALL': TokenType.ALL,
        'UNION': TokenType.UNION,
        'EXCEPT': TokenType.EXCEPT,
        'INTERSECT': TokenType.INTERSECT,
        'CASE': TokenType.CASE,
        'WHEN': TokenType.WHEN,
        'THEN': TokenType.THEN,
        'ELSE': TokenType.ELSE,
        'END': TokenType.END,
        'CAST': TokenType.CAST,
        'TRUE': TokenType.TRUE,
        'FALSE': TokenType.FALSE,
    }
    
    def __init__(self, sql: str):
        self.sql = sql
        self.pos = 0
        self.line = 1
        self.column = 1
        
    def tokenize(self) -> List[Token]:
        """Tokenize the SQL string."""
        tokens = []
        
        while self.pos < len(self.sql):
            # Skip whitespace
            if self.sql[self.pos].isspace():
                if self.sql[self.pos] == '\n':
                    self.line += 1
                    self.column = 1
                else:
                    self.column += 1
                self.pos += 1
                continue
                
            # Skip comments
            if self.pos + 1 < len(self.sql) and self.sql[self.pos:self.pos+2] == '--':
                while self.pos < len(self.sql) and self.sql[self.pos] != '\n':
                    self.pos += 1
                continue
                
            token = self._next_token()
            if token:
                tokens.append(token)
                
        tokens.append(Token(TokenType.EOF, '', self.line, self.column))
        return tokens
        
    def _next_token(self) -> Optional[Token]:
        """Get the next token."""
        start_line = self.line
        start_col = self.column
        
        char = self.sql[self.pos]
        
        # String literal
        if char in ('"', "'"):
            return self._read_string(start_line, start_col)
            
        # Number
        if char.isdigit() or (char == '.' and self.pos + 1 < len(self.sql) 
                              and self.sql[self.pos + 1].isdigit()):
            return self._read_number(start_line, start_col)
            
        # Identifier or keyword
        if char.isalpha() or char == '_':
            return self._read_identifier(start_line, start_col)
            
        # Operators and punctuation
        return self._read_operator(start_line, start_col)
        
    def _read_string(self, line: int, col: int) -> Token:
        """Read a string literal."""
        quote = self.sql[self.pos]
        self.pos += 1
        self.column += 1
        
        value = []
        while self.pos < len(self.sql) and self.sql[self.pos] != quote:
            if self.sql[self.pos] == '\\' and self.pos + 1 < len(self.sql):
                self.pos += 1
                self.column += 1
            value.append(self.sql[self.pos])
            self.pos += 1
            self.column += 1
            
        if self.pos < len(self.sql):
            self.pos += 1  # Skip closing quote
            self.column += 1
            
        return Token(TokenType.STRING, ''.join(value), line, col)
        
    def _read_number(self, line: int, col: int) -> Token:
        """Read a numeric literal."""
        value = []
        has_dot = False
        
        while self.pos < len(self.sql):
            char = self.sql[self.pos]
            if char.isdigit():
                value.append(char)
            elif char == '.' and not has_dot:
                value.append(char)
                has_dot = True
            elif char.lower() == 'e' and value:
                value.append(char)
                self.pos += 1
                self.column += 1
                if self.pos < len(self.sql) and self.sql[self.pos] in '+-':
                    value.append(self.sql[self.pos])
                    self.pos += 1
                    self.column += 1
                continue
            else:
                break
            self.pos += 1
            self.column += 1
            
        return Token(TokenType.NUMBER, ''.join(value), line, col)
        
    def _read_identifier(self, line: int, col: int) -> Token:
        """Read an identifier or keyword."""
        value = []
        
        while self.pos < len(self.sql):
            char = self.sql[self.pos]
            if char.isalnum() or char == '_':
                value.append(char)
                self.pos += 1
                self.column += 1
            else:
                break
                
        word = ''.join(value)
        upper = word.upper()
        
        if upper in self.KEYWORDS:
            return Token(self.KEYWORDS[upper], word, line, col)
        return Token(TokenType.IDENTIFIER, word, line, col)
        
    def _read_operator(self, line: int, col: int) -> Token:
        """Read an operator or punctuation."""
        char = self.sql[self.pos]
        self.pos += 1
        self.column += 1
        
        # Two-character operators
        if self.pos < len(self.sql):
            two_char = char + self.sql[self.pos]
            if two_char == '!=':
                self.pos += 1
                self.column += 1
                return Token(TokenType.NOT_EQUALS, two_char, line, col)
            if two_char == '<>':
                self.pos += 1
                self.column += 1
                return Token(TokenType.NOT_EQUALS, two_char, line, col)
            if two_char == '<=':
                self.pos += 1
                self.column += 1
                return Token(TokenType.LESS_EQUALS, two_char, line, col)
            if two_char == '>=':
                self.pos += 1
                self.column += 1
                return Token(TokenType.GREATER_EQUALS, two_char, line, col)
                
        # Single-character operators
        operators = {
            '=': TokenType.EQUALS,
            '<': TokenType.LESS_THAN,
            '>': TokenType.GREATER_THAN,
            '+': TokenType.PLUS,
            '-': TokenType.MINUS,
            '*': TokenType.STAR,
            '/': TokenType.SLASH,
            '%': TokenType.PERCENT,
            ',': TokenType.COMMA,
            '.': TokenType.DOT,
            '(': TokenType.LPAREN,
            ')': TokenType.RPAREN,
            ';': TokenType.SEMICOLON,
        }
        
        if char in operators:
            return Token(operators[char], char, line, col)
            
        raise ValueError(f"Unexpected character '{char}' at line {line}, column {col}")


# AST Node types
@dataclass
class ASTNode:
    """Base AST node."""
    pass


@dataclass
class SelectColumn(ASTNode):
    """A column in SELECT clause."""
    expression: 'Expression'
    alias: Optional[str] = None


@dataclass
class Expression(ASTNode):
    """Base expression node."""
    pass


@dataclass
class ColumnRef(Expression):
    """Column reference."""
    name: str
    table: Optional[str] = None


@dataclass
class Literal(Expression):
    """Literal value."""
    value: Any
    datatype: str = "unknown"


@dataclass
class FunctionCall(Expression):
    """Function call expression."""
    name: str
    args: List[Expression]
    distinct: bool = False


@dataclass
class BinaryOp(Expression):
    """Binary operation."""
    op: str
    left: Expression
    right: Expression


@dataclass
class UnaryOp(Expression):
    """Unary operation."""
    op: str
    operand: Expression


@dataclass
class BetweenExpr(Expression):
    """BETWEEN expression."""
    expr: Expression
    low: Expression
    high: Expression
    negated: bool = False


@dataclass
class InExpr(Expression):
    """IN expression."""
    expr: Expression
    values: List[Expression]
    negated: bool = False


@dataclass
class CaseExpr(Expression):
    """CASE expression."""
    operand: Optional[Expression]
    when_clauses: List[Tuple[Expression, Expression]]
    else_clause: Optional[Expression]


@dataclass 
class TableRef(ASTNode):
    """Table reference."""
    name: str
    alias: Optional[str] = None


@dataclass
class JoinClause(ASTNode):
    """JOIN clause."""
    type: str  # INNER, LEFT, RIGHT, FULL, CROSS
    table: TableRef
    condition: Optional[Expression] = None


@dataclass
class OrderByItem(ASTNode):
    """ORDER BY item."""
    expression: Expression
    ascending: bool = True


@dataclass
class SQLStatement(ASTNode):
    """Complete SQL statement."""
    select_columns: List[SelectColumn]
    from_table: Optional[TableRef] = None
    joins: List[JoinClause] = field(default_factory=list)
    where_clause: Optional[Expression] = None
    group_by: List[Expression] = field(default_factory=list)
    having_clause: Optional[Expression] = None
    order_by: List[OrderByItem] = field(default_factory=list)
    limit: Optional[int] = None
    offset: Optional[int] = None
    distinct: bool = False


class SQLParser:
    """
    SQL Parser
    ----------
    Recursive descent parser for SQL statements.
    """
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        
    def parse(self) -> SQLStatement:
        """Parse tokens into SQL statement."""
        return self._parse_select()
        
    def _current(self) -> Token:
        """Get current token."""
        return self.tokens[self.pos]
        
    def _peek(self, offset: int = 1) -> Token:
        """Peek at token."""
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return self.tokens[-1]
        
    def _advance(self) -> Token:
        """Advance to next token."""
        token = self._current()
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return token
        
    def _expect(self, token_type: TokenType) -> Token:
        """Expect a specific token type."""
        token = self._current()
        if token.type != token_type:
            raise ValueError(
                f"Expected {token_type.name} but got {token.type.name} "
                f"at line {token.line}, column {token.column}"
            )
        return self._advance()
        
    def _match(self, *types: TokenType) -> bool:
        """Check if current token matches any of the types."""
        return self._current().type in types
        
    def _parse_select(self) -> SQLStatement:
        """Parse SELECT statement."""
        self._expect(TokenType.SELECT)
        
        # Check for DISTINCT
        distinct = False
        if self._match(TokenType.DISTINCT):
            self._advance()
            distinct = True
            
        # Parse select columns
        select_columns = self._parse_select_columns()
        
        # Parse FROM
        from_table = None
        joins = []
        if self._match(TokenType.FROM):
            self._advance()
            from_table = self._parse_table_ref()
            
            # Parse JOINs
            while self._match(TokenType.JOIN, TokenType.INNER, TokenType.LEFT, 
                            TokenType.RIGHT, TokenType.FULL, TokenType.CROSS):
                joins.append(self._parse_join())
                
        # Parse WHERE
        where_clause = None
        if self._match(TokenType.WHERE):
            self._advance()
            where_clause = self._parse_expression()
            
        # Parse GROUP BY
        group_by = []
        if self._match(TokenType.GROUP):
            self._advance()
            self._expect(TokenType.BY)
            group_by = self._parse_expression_list()
            
        # Parse HAVING
        having_clause = None
        if self._match(TokenType.HAVING):
            self._advance()
            having_clause = self._parse_expression()
            
        # Parse ORDER BY
        order_by = []
        if self._match(TokenType.ORDER):
            self._advance()
            self._expect(TokenType.BY)
            order_by = self._parse_order_by_items()
            
        # Parse LIMIT
        limit = None
        if self._match(TokenType.LIMIT):
            self._advance()
            limit = int(self._expect(TokenType.NUMBER).value)
            
        # Parse OFFSET
        offset = None
        if self._match(TokenType.OFFSET):
            self._advance()
            offset = int(self._expect(TokenType.NUMBER).value)
            
        return SQLStatement(
            select_columns=select_columns,
            from_table=from_table,
            joins=joins,
            where_clause=where_clause,
            group_by=group_by,
            having_clause=having_clause,
            order_by=order_by,
            limit=limit,
            offset=offset,
            distinct=distinct
        )
        
    def _parse_select_columns(self) -> List[SelectColumn]:
        """Parse SELECT column list."""
        columns = []
        
        # Handle SELECT *
        if self._match(TokenType.STAR):
            self._advance()
            columns.append(SelectColumn(ColumnRef('*')))
            return columns
            
        columns.append(self._parse_select_column())
        
        while self._match(TokenType.COMMA):
            self._advance()
            columns.append(self._parse_select_column())
            
        return columns
        
    def _parse_select_column(self) -> SelectColumn:
        """Parse a single SELECT column."""
        expr = self._parse_expression()
        
        alias = None
        if self._match(TokenType.AS):
            self._advance()
            alias = self._expect(TokenType.IDENTIFIER).value
        elif self._match(TokenType.IDENTIFIER):
            # Implicit alias
            alias = self._advance().value
            
        return SelectColumn(expr, alias)
        
    def _parse_table_ref(self) -> TableRef:
        """Parse table reference."""
        name = self._expect(TokenType.IDENTIFIER).value
        
        alias = None
        if self._match(TokenType.AS):
            self._advance()
            alias = self._expect(TokenType.IDENTIFIER).value
        elif self._match(TokenType.IDENTIFIER):
            alias = self._advance().value
            
        return TableRef(name, alias)
        
    def _parse_join(self) -> JoinClause:
        """Parse JOIN clause."""
        join_type = 'INNER'
        
        if self._match(TokenType.LEFT):
            self._advance()
            join_type = 'LEFT'
            if self._match(TokenType.OUTER):
                self._advance()
        elif self._match(TokenType.RIGHT):
            self._advance()
            join_type = 'RIGHT'
            if self._match(TokenType.OUTER):
                self._advance()
        elif self._match(TokenType.FULL):
            self._advance()
            join_type = 'FULL'
            if self._match(TokenType.OUTER):
                self._advance()
        elif self._match(TokenType.CROSS):
            self._advance()
            join_type = 'CROSS'
        elif self._match(TokenType.INNER):
            self._advance()
            
        self._expect(TokenType.JOIN)
        table = self._parse_table_ref()
        
        condition = None
        if self._match(TokenType.ON):
            self._advance()
            condition = self._parse_expression()
            
        return JoinClause(join_type, table, condition)
        
    def _parse_expression_list(self) -> List[Expression]:
        """Parse comma-separated expressions."""
        exprs = [self._parse_expression()]
        
        while self._match(TokenType.COMMA):
            self._advance()
            exprs.append(self._parse_expression())
            
        return exprs
        
    def _parse_order_by_items(self) -> List[OrderByItem]:
        """Parse ORDER BY items."""
        items = [self._parse_order_by_item()]
        
        while self._match(TokenType.COMMA):
            self._advance()
            items.append(self._parse_order_by_item())
            
        return items
        
    def _parse_order_by_item(self) -> OrderByItem:
        """Parse single ORDER BY item."""
        expr = self._parse_expression()
        
        ascending = True
        if self._match(TokenType.ASC):
            self._advance()
        elif self._match(TokenType.DESC):
            self._advance()
            ascending = False
            
        return OrderByItem(expr, ascending)
        
    def _parse_expression(self) -> Expression:
        """Parse expression."""
        return self._parse_or_expr()
        
    def _parse_or_expr(self) -> Expression:
        """Parse OR expression."""
        left = self._parse_and_expr()
        
        while self._match(TokenType.OR):
            self._advance()
            right = self._parse_and_expr()
            left = BinaryOp('OR', left, right)
            
        return left
        
    def _parse_and_expr(self) -> Expression:
        """Parse AND expression."""
        left = self._parse_not_expr()
        
        while self._match(TokenType.AND):
            self._advance()
            right = self._parse_not_expr()
            left = BinaryOp('AND', left, right)
            
        return left
        
    def _parse_not_expr(self) -> Expression:
        """Parse NOT expression."""
        if self._match(TokenType.NOT):
            self._advance()
            return UnaryOp('NOT', self._parse_not_expr())
        return self._parse_comparison()
        
    def _parse_comparison(self) -> Expression:
        """Parse comparison expression."""
        left = self._parse_additive()
        
        # Handle IS NULL / IS NOT NULL
        if self._match(TokenType.IS):
            self._advance()
            negated = self._match(TokenType.NOT)
            if negated:
                self._advance()
            self._expect(TokenType.NULL)
            op = 'IS NOT NULL' if negated else 'IS NULL'
            return UnaryOp(op, left)
            
        # Handle BETWEEN
        if self._match(TokenType.BETWEEN):
            self._advance()
            low = self._parse_additive()
            self._expect(TokenType.AND)
            high = self._parse_additive()
            return BetweenExpr(left, low, high)
            
        # Handle IN
        if self._match(TokenType.IN):
            self._advance()
            self._expect(TokenType.LPAREN)
            values = self._parse_expression_list()
            self._expect(TokenType.RPAREN)
            return InExpr(left, values)
            
        # Handle LIKE
        if self._match(TokenType.LIKE):
            self._advance()
            right = self._parse_additive()
            return BinaryOp('LIKE', left, right)
            
        # Handle comparison operators
        comparison_ops = {
            TokenType.EQUALS: '=',
            TokenType.NOT_EQUALS: '!=',
            TokenType.LESS_THAN: '<',
            TokenType.LESS_EQUALS: '<=',
            TokenType.GREATER_THAN: '>',
            TokenType.GREATER_EQUALS: '>=',
        }
        
        if self._current().type in comparison_ops:
            op = comparison_ops[self._advance().type]
            right = self._parse_additive()
            return BinaryOp(op, left, right)
            
        return left
        
    def _parse_additive(self) -> Expression:
        """Parse additive expression (+, -)."""
        left = self._parse_multiplicative()
        
        while self._match(TokenType.PLUS, TokenType.MINUS):
            op = '+' if self._advance().type == TokenType.PLUS else '-'
            right = self._parse_multiplicative()
            left = BinaryOp(op, left, right)
            
        return left
        
    def _parse_multiplicative(self) -> Expression:
        """Parse multiplicative expression (*, /, %)."""
        left = self._parse_unary()
        
        while self._match(TokenType.STAR, TokenType.SLASH, TokenType.PERCENT):
            token = self._advance()
            if token.type == TokenType.STAR:
                op = '*'
            elif token.type == TokenType.SLASH:
                op = '/'
            else:
                op = '%'
            right = self._parse_unary()
            left = BinaryOp(op, left, right)
            
        return left
        
    def _parse_unary(self) -> Expression:
        """Parse unary expression."""
        if self._match(TokenType.MINUS):
            self._advance()
            return UnaryOp('-', self._parse_primary())
        return self._parse_primary()
        
    def _parse_primary(self) -> Expression:
        """Parse primary expression."""
        token = self._current()
        
        # Literal number
        if self._match(TokenType.NUMBER):
            self._advance()
            value = float(token.value) if '.' in token.value else int(token.value)
            return Literal(value, 'number')
            
        # Literal string
        if self._match(TokenType.STRING):
            self._advance()
            return Literal(token.value, 'string')
            
        # Boolean literals
        if self._match(TokenType.TRUE):
            self._advance()
            return Literal(True, 'boolean')
        if self._match(TokenType.FALSE):
            self._advance()
            return Literal(False, 'boolean')
            
        # NULL
        if self._match(TokenType.NULL):
            self._advance()
            return Literal(None, 'null')
            
        # CASE expression
        if self._match(TokenType.CASE):
            return self._parse_case()
            
        # Parenthesized expression
        if self._match(TokenType.LPAREN):
            self._advance()
            expr = self._parse_expression()
            self._expect(TokenType.RPAREN)
            return expr
            
        # Star (for COUNT(*))
        if self._match(TokenType.STAR):
            self._advance()
            return ColumnRef('*')
            
        # Identifier (column or function)
        if self._match(TokenType.IDENTIFIER):
            name = self._advance().value
            
            # Check for function call
            if self._match(TokenType.LPAREN):
                return self._parse_function_call(name)
                
            # Check for qualified name (table.column)
            if self._match(TokenType.DOT):
                self._advance()
                col_name = self._expect(TokenType.IDENTIFIER).value
                return ColumnRef(col_name, name)
                
            return ColumnRef(name)
            
        raise ValueError(
            f"Unexpected token {token.type.name} at line {token.line}, column {token.column}"
        )
        
    def _parse_function_call(self, name: str) -> FunctionCall:
        """Parse function call."""
        self._expect(TokenType.LPAREN)
        
        args = []
        distinct = False
        
        # Check for DISTINCT
        if self._match(TokenType.DISTINCT):
            self._advance()
            distinct = True
            
        if not self._match(TokenType.RPAREN):
            args = self._parse_expression_list()
            
        self._expect(TokenType.RPAREN)
        
        return FunctionCall(name.upper(), args, distinct)
        
    def _parse_case(self) -> CaseExpr:
        """Parse CASE expression."""
        self._expect(TokenType.CASE)
        
        # Check for simple CASE (with operand)
        operand = None
        if not self._match(TokenType.WHEN):
            operand = self._parse_expression()
            
        when_clauses = []
        while self._match(TokenType.WHEN):
            self._advance()
            condition = self._parse_expression()
            self._expect(TokenType.THEN)
            result = self._parse_expression()
            when_clauses.append((condition, result))
            
        else_clause = None
        if self._match(TokenType.ELSE):
            self._advance()
            else_clause = self._parse_expression()
            
        self._expect(TokenType.END)
        
        return CaseExpr(operand, when_clauses, else_clause)
