"""
HiveFrame Complex Event Processing (CEP)
========================================
Pattern detection in streaming data using bee-inspired optimization.

Key Features:
- Pattern matching across event sequences
- Temporal pattern detection with windowing
- Complex condition evaluation
- Pattern aggregation and correlation
- Bee-inspired pattern optimization
"""

import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar

from ..core import ColonyState, WaggleDance


class QuantifierType(Enum):
    """Pattern quantifier types."""

    ONE = "one"  # Exactly one occurrence
    ONE_OR_MORE = "one_or_more"  # At least one occurrence
    ZERO_OR_MORE = "zero_or_more"  # Any number of occurrences
    OPTIONAL = "optional"  # Zero or one occurrence
    TIMES = "times"  # Specific number of times
    TIMES_OR_MORE = "times_or_more"  # At least N times


class ContiguityType(Enum):
    """Pattern contiguity constraints."""

    STRICT = "strict"  # Events must be strictly consecutive
    RELAXED = "relaxed"  # Allow gaps between events
    NON_DETERMINISTIC = "non_deterministic"  # Explore all possibilities


@dataclass
class PatternCondition:
    """A condition that an event must satisfy."""

    field: str
    operator: str  # eq, ne, gt, lt, gte, lte, contains, regex
    value: Any
    negate: bool = False

    def evaluate(self, event: Dict[str, Any]) -> bool:
        """Evaluate this condition against an event."""
        if self.field not in event:
            result = False
        else:
            field_value = event[self.field]
            result = self._compare(field_value)

        return not result if self.negate else result

    def _compare(self, field_value: Any) -> bool:
        """Compare field value with condition value."""
        if self.operator == "eq":
            return bool(field_value == self.value)
        elif self.operator == "ne":
            return bool(field_value != self.value)
        elif self.operator == "gt":
            return bool(field_value > self.value)
        elif self.operator == "lt":
            return bool(field_value < self.value)
        elif self.operator == "gte":
            return bool(field_value >= self.value)
        elif self.operator == "lte":
            return bool(field_value <= self.value)
        elif self.operator == "contains":
            return self.value in str(field_value)
        elif self.operator == "regex":
            return bool(re.match(str(self.value), str(field_value)))
        elif self.operator == "in":
            return bool(field_value in self.value)
        elif self.operator == "not_in":
            return bool(field_value not in self.value)
        else:
            return False


@dataclass
class PatternState:
    """
    Represents the state of a pattern being matched.

    Tracks matched events and progress through pattern.
    """

    pattern_id: str
    current_stage: int = 0
    matched_events: List[Dict[str, Any]] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    match_count: int = 0  # For quantifiers
    is_complete: bool = False
    is_timed_out: bool = False

    def clone(self) -> "PatternState":
        """Create a copy of this state for branching."""
        return PatternState(
            pattern_id=self.pattern_id,
            current_stage=self.current_stage,
            matched_events=list(self.matched_events),
            start_time=self.start_time,
            match_count=self.match_count,
            is_complete=self.is_complete,
            is_timed_out=self.is_timed_out,
        )


@dataclass
class PatternMatch:
    """A complete pattern match result."""

    pattern_id: str
    events: List[Dict[str, Any]]
    start_time: float
    end_time: float
    match_duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class PatternElement:
    """
    A single element in a pattern definition.

    Represents one step in the pattern with conditions and quantifiers.
    """

    def __init__(
        self,
        name: str,
        conditions: Optional[List[PatternCondition]] = None,
        quantifier: QuantifierType = QuantifierType.ONE,
        quantifier_value: int = 1,
    ):
        self.name = name
        self.conditions = conditions or []
        self.quantifier = quantifier
        self.quantifier_value = quantifier_value
        self.strict = False  # Whether this element requires strict contiguity

    def matches(self, event: Dict[str, Any]) -> bool:
        """Check if an event matches this element's conditions."""
        if not self.conditions:
            return True
        return all(cond.evaluate(event) for cond in self.conditions)

    def where(self, field: str, operator: str, value: Any) -> "PatternElement":
        """Add a condition to this element (fluent API)."""
        self.conditions.append(PatternCondition(field, operator, value))
        return self

    def times(self, n: int) -> "PatternElement":
        """Require exactly n occurrences."""
        self.quantifier = QuantifierType.TIMES
        self.quantifier_value = n
        return self

    def one_or_more(self) -> "PatternElement":
        """Require at least one occurrence."""
        self.quantifier = QuantifierType.ONE_OR_MORE
        return self

    def optional(self) -> "PatternElement":
        """Make this element optional."""
        self.quantifier = QuantifierType.OPTIONAL
        return self


class Pattern:
    """
    Fluent API for defining CEP patterns.

    Usage:
        pattern = (
            Pattern("fraud_detection")
            .begin("login").where("type", "eq", "login")
            .followed_by("withdrawal").where("type", "eq", "withdrawal")
                                      .where("amount", "gt", 10000)
            .within(minutes=5)
        )
    """

    def __init__(self, name: str):
        self.name = name
        self.elements: List[PatternElement] = []
        self.contiguity: ContiguityType = ContiguityType.RELAXED
        self.timeout_seconds: Optional[float] = None
        self._current_element: Optional[PatternElement] = None

    def begin(self, element_name: str) -> "Pattern":
        """Start the pattern with an element."""
        element = PatternElement(element_name)
        self.elements.append(element)
        self._current_element = element
        return self

    def followed_by(self, element_name: str) -> "Pattern":
        """Add a relaxed contiguity element."""
        element = PatternElement(element_name)
        self.elements.append(element)
        self._current_element = element
        return self

    def next(self, element_name: str) -> "Pattern":
        """Add a strict contiguity element."""
        element = PatternElement(element_name)
        element.strict = True
        self.elements.append(element)
        self._current_element = element
        return self

    def where(self, field: str, operator: str, value: Any) -> "Pattern":
        """Add a condition to the current element."""
        if self._current_element:
            self._current_element.conditions.append(PatternCondition(field, operator, value))
        return self

    def times(self, n: int) -> "Pattern":
        """Set the quantifier for the current element."""
        if self._current_element:
            self._current_element.times(n)
        return self

    def one_or_more(self) -> "Pattern":
        """Require at least one occurrence of current element."""
        if self._current_element:
            self._current_element.one_or_more()
        return self

    def optional(self) -> "Pattern":
        """Make the current element optional."""
        if self._current_element:
            self._current_element.optional()
        return self

    def within(
        self,
        seconds: float = 0,
        minutes: float = 0,
        hours: float = 0,
    ) -> "Pattern":
        """Set a time window for the pattern."""
        self.timeout_seconds = seconds + (minutes * 60) + (hours * 3600)
        return self

    def strict_contiguity(self) -> "Pattern":
        """Require strict contiguity between all elements."""
        self.contiguity = ContiguityType.STRICT
        return self

    def relaxed_contiguity(self) -> "Pattern":
        """Allow gaps between pattern elements."""
        self.contiguity = ContiguityType.RELAXED
        return self


class PatternMatcher:
    """
    NFA-based pattern matcher for event streams.

    Uses Non-deterministic Finite Automaton for efficient pattern matching.
    """

    def __init__(self, pattern: Pattern):
        self.pattern = pattern
        self.active_states: List[PatternState] = []
        self._matches: List[PatternMatch] = []
        self._lock_states = False

    def process_event(
        self, event: Dict[str, Any], event_time: Optional[float] = None
    ) -> List[PatternMatch]:
        """
        Process an event and return any completed matches.

        Args:
            event: The event to process
            event_time: Optional event timestamp (defaults to current time)

        Returns:
            List of completed pattern matches
        """
        current_time = event_time or time.time()
        new_matches: List[PatternMatch] = []
        new_states: List[PatternState] = []

        # Check for timeouts
        for state in self.active_states:
            if self.pattern.timeout_seconds:
                if current_time - state.start_time > self.pattern.timeout_seconds:
                    state.is_timed_out = True
                    continue

            # Try to advance this state
            advanced_states = self._try_advance(state, event, current_time)
            new_states.extend(advanced_states)

        # Try to start a new pattern match
        if self.pattern.elements:
            first_element = self.pattern.elements[0]
            if first_element.matches(event):
                new_state = PatternState(
                    pattern_id=self.pattern.name,
                    current_stage=0,
                    matched_events=[event],
                    start_time=current_time,
                    match_count=1,
                )
                # Check if single-element pattern is complete
                if self._check_element_complete(first_element, new_state):
                    if len(self.pattern.elements) == 1:
                        new_state.is_complete = True
                    else:
                        new_state.current_stage = 1
                        new_state.match_count = 0

                new_states.append(new_state)

        # Filter completed states and extract matches
        self.active_states = []
        for state in new_states:
            if state.is_complete:
                match = PatternMatch(
                    pattern_id=state.pattern_id,
                    events=state.matched_events,
                    start_time=state.start_time,
                    end_time=current_time,
                    match_duration=current_time - state.start_time,
                )
                new_matches.append(match)
                self._matches.append(match)
            elif not state.is_timed_out:
                self.active_states.append(state)

        return new_matches

    def _try_advance(
        self,
        state: PatternState,
        event: Dict[str, Any],
        current_time: float,
    ) -> List[PatternState]:
        """Try to advance a pattern state with the given event."""
        result_states: List[PatternState] = []

        if state.current_stage >= len(self.pattern.elements):
            return result_states

        current_element = self.pattern.elements[state.current_stage]

        # Check if event matches current element
        if current_element.matches(event):
            # Clone state and add matched event
            new_state = state.clone()
            new_state.matched_events.append(event)
            new_state.match_count += 1

            # Check if element is complete (satisfied quantifier)
            if self._check_element_complete(current_element, new_state):
                if new_state.current_stage + 1 >= len(self.pattern.elements):
                    # Pattern complete
                    new_state.is_complete = True
                else:
                    # Move to next element
                    new_state.current_stage += 1
                    new_state.match_count = 0

            result_states.append(new_state)

            # For non-deterministic matching, also keep the current state
            if self.pattern.contiguity == ContiguityType.NON_DETERMINISTIC:
                result_states.append(state)

        # For relaxed contiguity, keep state even if event doesn't match
        elif self.pattern.contiguity in (
            ContiguityType.RELAXED,
            ContiguityType.NON_DETERMINISTIC,
        ):
            result_states.append(state)

        return result_states

    def _check_element_complete(self, element: PatternElement, state: PatternState) -> bool:
        """Check if an element's quantifier is satisfied."""
        if element.quantifier == QuantifierType.ONE:
            return state.match_count >= 1
        elif element.quantifier == QuantifierType.ONE_OR_MORE:
            return state.match_count >= 1
        elif element.quantifier == QuantifierType.ZERO_OR_MORE:
            return True
        elif element.quantifier == QuantifierType.OPTIONAL:
            return True
        elif element.quantifier == QuantifierType.TIMES:
            return state.match_count >= element.quantifier_value
        elif element.quantifier == QuantifierType.TIMES_OR_MORE:
            return state.match_count >= element.quantifier_value
        return True

    def get_active_count(self) -> int:
        """Get number of active pattern states."""
        return len(self.active_states)

    def get_matches(self) -> List[PatternMatch]:
        """Get all completed matches."""
        return self._matches.copy()

    def clear_matches(self) -> None:
        """Clear the list of completed matches."""
        self._matches.clear()


T = TypeVar("T")


class CEPEngine:
    """
    Complex Event Processing Engine with bee-inspired optimization.

    Processes event streams against multiple patterns simultaneously,
    using swarm intelligence for optimal resource allocation.

    Usage:
        engine = CEPEngine()
        engine.add_pattern(
            Pattern("login_failure")
            .begin("login").where("success", "eq", False)
            .times(3)
            .within(minutes=5)
        )
        engine.add_callback("login_failure", alert_security)

        for event in events:
            matches = engine.process_event(event)
    """

    def __init__(
        self,
        max_active_states: int = 10000,
        enable_optimization: bool = True,
    ):
        self.max_active_states = max_active_states
        self.enable_optimization = enable_optimization

        self.colony = ColonyState()
        self._patterns: Dict[str, Pattern] = {}
        self._matchers: Dict[str, PatternMatcher] = {}
        self._callbacks: Dict[str, List[Callable[[PatternMatch], None]]] = defaultdict(list)

        # Metrics
        self._events_processed = 0
        self._matches_found = 0
        self._processing_times: deque = deque(maxlen=1000)

    def add_pattern(self, pattern: Pattern) -> None:
        """Add a pattern to the engine."""
        self._patterns[pattern.name] = pattern
        self._matchers[pattern.name] = PatternMatcher(pattern)

    def remove_pattern(self, pattern_name: str) -> None:
        """Remove a pattern from the engine."""
        self._patterns.pop(pattern_name, None)
        self._matchers.pop(pattern_name, None)
        self._callbacks.pop(pattern_name, None)

    def add_callback(
        self,
        pattern_name: str,
        callback: Callable[[PatternMatch], None],
    ) -> None:
        """Add a callback to be invoked when a pattern matches."""
        self._callbacks[pattern_name].append(callback)

    def process_event(
        self, event: Dict[str, Any], event_time: Optional[float] = None
    ) -> List[PatternMatch]:
        """
        Process an event against all registered patterns.

        Returns all matches found.
        """
        start_time = time.time()
        all_matches: List[PatternMatch] = []

        for pattern_name, matcher in self._matchers.items():
            matches = matcher.process_event(event, event_time)
            all_matches.extend(matches)

            # Invoke callbacks
            for match in matches:
                for callback in self._callbacks.get(pattern_name, []):
                    try:
                        callback(match)
                    except Exception:
                        pass  # Don't let callback errors break processing

        # Record metrics
        processing_time = time.time() - start_time
        self._events_processed += 1
        self._matches_found += len(all_matches)
        self._processing_times.append(processing_time)

        # Perform waggle dance for fitness tracking
        if self.enable_optimization:
            dance = WaggleDance(
                partition_id="cep_engine",
                quality_score=1.0 if not all_matches else 1.0 + len(all_matches) * 0.1,
                processing_time=processing_time,
                result_size=len(all_matches),
                worker_id="cep_engine",
            )
            self.colony.dance_floor.perform_dance(dance)

        return all_matches

    def process_events(self, events: List[Dict[str, Any]]) -> List[PatternMatch]:
        """Process a batch of events."""
        all_matches: List[PatternMatch] = []
        for event in events:
            matches = self.process_event(event)
            all_matches.extend(matches)
        return all_matches

    def get_active_states_count(self) -> Dict[str, int]:
        """Get count of active states per pattern."""
        return {name: matcher.get_active_count() for name, matcher in self._matchers.items()}

    def get_metrics(self) -> Dict[str, Any]:
        """Get engine metrics."""
        avg_time = (
            sum(self._processing_times) / len(self._processing_times)
            if self._processing_times
            else 0
        )
        return {
            "events_processed": self._events_processed,
            "matches_found": self._matches_found,
            "patterns_registered": len(self._patterns),
            "total_active_states": sum(self.get_active_states_count().values()),
            "avg_processing_time_ms": avg_time * 1000,
            "active_states_per_pattern": self.get_active_states_count(),
        }

    def reset(self) -> None:
        """Reset all pattern matchers."""
        for matcher in self._matchers.values():
            matcher.active_states.clear()
            matcher.clear_matches()


# Convenience functions for creating patterns
def pattern(name: str) -> Pattern:
    """Create a new pattern with the given name."""
    return Pattern(name)


def begin(element_name: str) -> PatternElement:
    """Create a pattern element."""
    return PatternElement(element_name)


def condition(field: str, operator: str, value: Any, negate: bool = False) -> PatternCondition:
    """Create a pattern condition."""
    return PatternCondition(field, operator, value, negate)
