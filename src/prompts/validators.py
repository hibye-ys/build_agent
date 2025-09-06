"""Validators and optimization tools for prompts."""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import re
import tiktoken
import logging
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of prompt validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, error: str):
        """Add an error message."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add a warning message."""
        self.warnings.append(warning)
    
    def add_suggestion(self, suggestion: str):
        """Add a suggestion."""
        self.suggestions.append(suggestion)


class PromptValidator:
    """Validates prompt templates for common issues."""
    
    def __init__(
        self,
        max_length: int = 4000,
        max_tokens: int = 2000,
        check_injection: bool = True,
        check_formatting: bool = True
    ):
        """Initialize validator.
        
        Args:
            max_length: Maximum character length
            max_tokens: Maximum token count
            check_injection: Whether to check for injection attacks
            check_formatting: Whether to check formatting issues
        """
        self.max_length = max_length
        self.max_tokens = max_tokens
        self.check_injection = check_injection
        self.check_formatting = check_formatting
    
    def validate(self, prompt: str, variables: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate a prompt template.
        
        Args:
            prompt: Prompt template string
            variables: Variables to check
            
        Returns:
            ValidationResult with findings
        """
        result = ValidationResult(is_valid=True)
        
        # Check length
        if len(prompt) > self.max_length:
            result.add_error(f"Prompt exceeds maximum length ({len(prompt)} > {self.max_length})")
        
        # Check token count
        token_count = self._estimate_tokens(prompt)
        result.metrics["token_count"] = token_count
        
        if token_count > self.max_tokens:
            result.add_error(f"Prompt exceeds maximum tokens ({token_count} > {self.max_tokens})")
        
        # Check for injection vulnerabilities
        if self.check_injection:
            self._check_injection_vulnerabilities(prompt, result)
        
        # Check formatting
        if self.check_formatting:
            self._check_formatting_issues(prompt, result)
        
        # Check variables
        if variables:
            self._check_variables(prompt, variables, result)
        
        # Add optimization suggestions
        self._add_optimization_suggestions(prompt, result)
        
        return result
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated token count
        """
        try:
            # Try to use tiktoken for accurate counting
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            # Fallback to simple estimation (1 token ≈ 4 characters)
            return len(text) // 4
    
    def _check_injection_vulnerabilities(self, prompt: str, result: ValidationResult):
        """Check for potential injection vulnerabilities.
        
        Args:
            prompt: Prompt to check
            result: Result to update
        """
        # Check for dangerous patterns
        dangerous_patterns = [
            r"(?i)ignore\s+previous\s+instructions",
            r"(?i)disregard\s+all\s+prior",
            r"(?i)forget\s+everything",
            r"(?i)system\s+prompt.*override",
            r"(?i)jailbreak",
            r"(?i)bypass\s+safety"
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, prompt):
                result.add_warning(f"Potential injection pattern detected: {pattern}")
        
        # Check for suspicious Unicode characters
        suspicious_chars = [
            '\u202e',  # Right-to-left override
            '\u200b',  # Zero-width space
            '\ufeff',  # Zero-width no-break space
        ]
        
        for char in suspicious_chars:
            if char in prompt:
                result.add_warning(f"Suspicious Unicode character detected: U+{ord(char):04X}")
    
    def _check_formatting_issues(self, prompt: str, result: ValidationResult):
        """Check for formatting issues.
        
        Args:
            prompt: Prompt to check
            result: Result to update
        """
        # Check for excessive whitespace
        if re.search(r'\n{4,}', prompt):
            result.add_warning("Excessive line breaks detected")
            result.add_suggestion("Consider reducing consecutive line breaks to maximum 2")
        
        if re.search(r'  {3,}', prompt):
            result.add_warning("Excessive spaces detected")
            result.add_suggestion("Consider reducing consecutive spaces")
        
        # Check for unbalanced brackets
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        
        for char in prompt:
            if char in brackets:
                stack.append(brackets[char])
            elif char in brackets.values():
                if not stack or stack.pop() != char:
                    result.add_warning("Unbalanced brackets detected")
                    break
        
        if stack:
            result.add_warning("Unclosed brackets detected")
        
        # Check for proper sentence ending
        if prompt.strip() and not prompt.strip()[-1] in '.!?:':
            result.add_suggestion("Consider ending prompt with proper punctuation")
    
    def _check_variables(self, prompt: str, variables: Dict[str, Any], result: ValidationResult):
        """Check variable usage in prompt.
        
        Args:
            prompt: Prompt template
            variables: Provided variables
            result: Result to update
        """
        # Find all Jinja2 variables in prompt
        var_pattern = r'\{\{\s*(\w+)(?:\.\w+)*\s*\}\}'
        found_vars = set(re.findall(var_pattern, prompt))
        
        # Check for missing variables
        provided_vars = set(variables.keys())
        missing_vars = found_vars - provided_vars
        
        if missing_vars:
            result.add_error(f"Missing required variables: {', '.join(missing_vars)}")
        
        # Check for unused variables
        unused_vars = provided_vars - found_vars
        
        if unused_vars:
            result.add_warning(f"Unused variables provided: {', '.join(unused_vars)}")
    
    def _add_optimization_suggestions(self, prompt: str, result: ValidationResult):
        """Add optimization suggestions.
        
        Args:
            prompt: Prompt to analyze
            result: Result to update
        """
        # Check for redundant phrases
        redundant_phrases = [
            ("please please", "please"),
            ("very very", "very"),
            ("really really", "really"),
            ("in order to", "to"),
            ("at this point in time", "now"),
            ("due to the fact that", "because")
        ]
        
        for phrase, replacement in redundant_phrases:
            if phrase.lower() in prompt.lower():
                result.add_suggestion(f"Consider replacing '{phrase}' with '{replacement}' to reduce tokens")
        
        # Check for verbose constructions
        if len(prompt.split()) > 500:
            result.add_suggestion("Consider breaking down this prompt into smaller, focused sections")
        
        # Check for repetitive instructions
        sentences = prompt.split('.')
        if len(sentences) > len(set(sentences)) * 1.5:
            result.add_suggestion("Consider consolidating repetitive instructions")


class TokenCounter:
    """Counts tokens for different models."""
    
    # Token limits for common models
    MODEL_LIMITS = {
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-4-turbo": 128000,
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-16k": 16384,
        "claude-3": 100000,
        "claude-2": 100000,
        "gemini-pro": 30720
    }
    
    # Cost per 1K tokens (input/output)
    MODEL_COSTS = {
        "gpt-4": (0.03, 0.06),
        "gpt-4-turbo": (0.01, 0.03),
        "gpt-3.5-turbo": (0.0005, 0.0015),
        "claude-3": (0.008, 0.024),
        "claude-2": (0.008, 0.024),
        "gemini-pro": (0.00025, 0.0005)
    }
    
    def __init__(self, model: str = "gpt-4"):
        """Initialize token counter.
        
        Args:
            model: Model to count tokens for
        """
        self.model = model
        self.encoding = None
        
        try:
            # Try to get model-specific encoding
            if "gpt" in model.lower():
                self.encoding = tiktoken.encoding_for_model(model)
            else:
                # Use default encoding for other models
                self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Could not load tokenizer: {e}")
    
    def count(self, text: str) -> int:
        """Count tokens in text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Token count
        """
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Fallback estimation
            return len(text) // 4
    
    def estimate_cost(
        self,
        input_text: str,
        expected_output_tokens: int = 500
    ) -> Tuple[float, float]:
        """Estimate cost for a prompt.
        
        Args:
            input_text: Input prompt
            expected_output_tokens: Expected output length
            
        Returns:
            Tuple of (input_cost, output_cost) in dollars
        """
        input_tokens = self.count(input_text)
        
        if self.model in self.MODEL_COSTS:
            input_rate, output_rate = self.MODEL_COSTS[self.model]
            input_cost = (input_tokens / 1000) * input_rate
            output_cost = (expected_output_tokens / 1000) * output_rate
            return input_cost, output_cost
        
        return 0.0, 0.0
    
    def check_limit(self, text: str) -> Tuple[bool, int, int]:
        """Check if text exceeds model token limit.
        
        Args:
            text: Text to check
            
        Returns:
            Tuple of (within_limit, token_count, limit)
        """
        token_count = self.count(text)
        limit = self.MODEL_LIMITS.get(self.model, 4096)
        
        return token_count <= limit, token_count, limit


class PromptOptimizer:
    """Optimizes prompts for token efficiency."""
    
    def __init__(self, target_reduction: float = 0.3):
        """Initialize optimizer.
        
        Args:
            target_reduction: Target reduction ratio (0.3 = 30% reduction)
        """
        self.target_reduction = target_reduction
        self.token_counter = TokenCounter()
    
    def optimize(self, prompt: str, preserve_meaning: bool = True) -> Tuple[str, Dict[str, Any]]:
        """Optimize a prompt for token efficiency.
        
        Args:
            prompt: Prompt to optimize
            preserve_meaning: Whether to preserve exact meaning
            
        Returns:
            Tuple of (optimized_prompt, metrics)
        """
        original_tokens = self.token_counter.count(prompt)
        optimized = prompt
        
        # Apply optimization techniques
        optimized = self._remove_redundancy(optimized)
        optimized = self._simplify_language(optimized)
        optimized = self._compress_whitespace(optimized)
        
        if not preserve_meaning:
            optimized = self._aggressive_compression(optimized)
        
        # Calculate metrics
        optimized_tokens = self.token_counter.count(optimized)
        reduction = 1 - (optimized_tokens / original_tokens)
        
        metrics = {
            "original_tokens": original_tokens,
            "optimized_tokens": optimized_tokens,
            "reduction_ratio": reduction,
            "target_met": reduction >= self.target_reduction
        }
        
        return optimized, metrics
    
    def _remove_redundancy(self, text: str) -> str:
        """Remove redundant phrases and words.
        
        Args:
            text: Text to process
            
        Returns:
            Processed text
        """
        # Remove redundant phrases
        redundant_replacements = [
            (r'\bvery\s+very\b', 'very'),
            (r'\breally\s+really\b', 'really'),
            (r'\bplease\s+please\b', 'please'),
            (r'\bin order to\b', 'to'),
            (r'\bat this point in time\b', 'now'),
            (r'\bdue to the fact that\b', 'because'),
            (r'\bin the event that\b', 'if'),
            (r'\bfor the purpose of\b', 'to'),
            (r'\bwith regard to\b', 'about'),
            (r'\bin spite of the fact that\b', 'although')
        ]
        
        for pattern, replacement in redundant_replacements:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _simplify_language(self, text: str) -> str:
        """Simplify verbose language.
        
        Args:
            text: Text to simplify
            
        Returns:
            Simplified text
        """
        # Simplify common verbose constructions
        simplifications = [
            (r'\bUtilize\b', 'Use'),
            (r'\bFacilitate\b', 'Help'),
            (r'\bEnumerate\b', 'List'),
            (r'\bCommence\b', 'Start'),
            (r'\bTerminate\b', 'End'),
            (r'\bImplement\b', 'Do'),
            (r'\bDemonstrate\b', 'Show'),
            (r'\bIndicate\b', 'Show'),
            (r'\bSubsequently\b', 'Then'),
            (r'\bPrior to\b', 'Before')
        ]
        
        for pattern, replacement in simplifications:
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def _compress_whitespace(self, text: str) -> str:
        """Compress excessive whitespace.
        
        Args:
            text: Text to compress
            
        Returns:
            Compressed text
        """
        # Replace multiple spaces with single space
        text = re.sub(r' {2,}', ' ', text)
        
        # Replace multiple newlines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove trailing whitespace
        lines = [line.rstrip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()
    
    def _aggressive_compression(self, text: str) -> str:
        """Apply aggressive compression techniques.
        
        Args:
            text: Text to compress
            
        Returns:
            Compressed text
        """
        # Remove articles where possible
        text = re.sub(r'\b(the|a|an)\s+', '', text, flags=re.IGNORECASE)
        
        # Use abbreviations
        abbreviations = [
            (r'\bfor example\b', 'e.g.'),
            (r'\bthat is\b', 'i.e.'),
            (r'\bet cetera\b', 'etc.'),
            (r'\bversus\b', 'vs.'),
            (r'\bapproximately\b', '~'),
            (r'\bequals\b', '='),
            (r'\bgreater than\b', '>'),
            (r'\bless than\b', '<')
        ]
        
        for pattern, replacement in abbreviations:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Remove unnecessary words
        unnecessary = [
            r'\bjust\b',
            r'\bsimply\b',
            r'\bbasically\b',
            r'\bactually\b',
            r'\breally\b',
            r'\bvery\b'
        ]
        
        for pattern in unnecessary:
            text = re.sub(pattern + r'\s*', '', text, flags=re.IGNORECASE)
        
        return text


class PromptComplexityAnalyzer:
    """Analyzes prompt complexity and readability."""
    
    def analyze(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt complexity.
        
        Args:
            prompt: Prompt to analyze
            
        Returns:
            Dictionary of complexity metrics
        """
        words = prompt.split()
        sentences = re.split(r'[.!?]+', prompt)
        sentences = [s for s in sentences if s.strip()]
        
        # Calculate basic metrics
        word_count = len(words)
        sentence_count = len(sentences)
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        # Calculate vocabulary diversity
        unique_words = set(word.lower() for word in words)
        vocabulary_diversity = len(unique_words) / max(word_count, 1)
        
        # Estimate reading level (simplified Flesch-Kincaid)
        syllable_count = sum(self._count_syllables(word) for word in words)
        avg_syllables_per_word = syllable_count / max(word_count, 1)
        
        reading_ease = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word
        reading_ease = max(0, min(100, reading_ease))
        
        # Determine complexity level
        if reading_ease >= 90:
            complexity = "very easy"
        elif reading_ease >= 80:
            complexity = "easy"
        elif reading_ease >= 70:
            complexity = "fairly easy"
        elif reading_ease >= 60:
            complexity = "standard"
        elif reading_ease >= 50:
            complexity = "fairly difficult"
        elif reading_ease >= 30:
            complexity = "difficult"
        else:
            complexity = "very difficult"
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_length,
            "vocabulary_diversity": vocabulary_diversity,
            "reading_ease_score": reading_ease,
            "complexity_level": complexity,
            "estimated_tokens": word_count // 0.75  # Rough estimate
        }
    
    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count for a word.
        
        Args:
            word: Word to count syllables for
            
        Returns:
            Estimated syllable count
        """
        word = word.lower()
        vowels = "aeiou"
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Adjust for silent e
        if word.endswith('e'):
            syllable_count -= 1
        
        # Ensure at least one syllable
        return max(1, syllable_count)