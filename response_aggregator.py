"""
Response Aggregator: Combines all pipeline outputs
Aggregates query expansion, topic classification, and bot response into single dict.
"""

from typing import Dict, Optional


class ResponseAggregator:
    """Aggregates outputs from query expander, classifier, and bot response."""
    
    @staticmethod
    def aggregate(
        original_query: str,
        expanded_query: str,
        level1_topic: str,
        level2_topic: str,
        confidence: float,
        bot_answer: str,
        is_general: bool = False,
        classification_error: Optional[str] = None,
        level1_confidence: float = None,
        level2_confidence: float = None
    ) -> Dict:
        """
        Aggregate all pipeline outputs into a single dictionary.
        
        Args:
            original_query: Original user query (before expansion)
            expanded_query: Expanded query from Claude
            level1_topic: Level 1 classification (vertical)
            level2_topic: Level 2 classification (subdomain)
            confidence: Classification confidence score
            level1_confidence: Optional Level 1 confidence score
            level2_confidence: Optional Level 2 confidence score
            bot_answer: The bot's response to the query
            is_general: Whether this was classified as general/chitchat
            classification_error: Any error from classification pipeline
        
        Returns:
            Aggregated result dictionary with all pipeline information
        """
        result = {
            "original_query": original_query,
            "expanded_query": expanded_query,
            "level1_topic": level1_topic,
            "level2_topic": level2_topic,
            "confidence": confidence,
            "level1_confidence": level1_confidence,
            "level2_confidence": level2_confidence,
            "bot_answer": bot_answer,
            "is_general": is_general,
            "classification_error": classification_error,
            "topic_tag": f"{level1_topic} > {level2_topic}"
        }
        
        return result
    
    @staticmethod
    def format_for_display(aggregated_result: Dict) -> Dict:
        """
        Format aggregated result for UI display.
        
        Args:
            aggregated_result: Result dictionary from aggregate()
        
        Returns:
            Formatted dictionary with UI-friendly strings
        """
        confidence_percent = round(aggregated_result["confidence"] * 100, 1)
        
        return {
            "original_query": aggregated_result["original_query"],
            "expanded_query": aggregated_result["expanded_query"],
            "topic_tag": aggregated_result["topic_tag"],
            "confidence_percent": confidence_percent,
            "level1_confidence_percent": round((aggregated_result.get("level1_confidence") or 0.0) * 100, 1),
            "level2_confidence_percent": round((aggregated_result.get("level2_confidence") or 0.0) * 100, 1),
            "bot_answer": aggregated_result["bot_answer"],
            "is_general": aggregated_result["is_general"],
            "error": aggregated_result["classification_error"]
        }
    
    @staticmethod
    def validate_result(result: Dict) -> bool:
        """
        Validate that aggregated result has all required fields.
        
        Args:
            result: Aggregated result dictionary
        
        Returns:
            True if valid, False otherwise
        """
        required_fields = [
            "original_query",
            "expanded_query",
            "level1_topic",
            "level2_topic",
            "confidence",
            "bot_answer",
            "is_general",
            "topic_tag"
        ]
        
        return all(field in result for field in required_fields)
