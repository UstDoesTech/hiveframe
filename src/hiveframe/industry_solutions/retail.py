"""
HiveFrame for Retail - Customer 360 and demand forecasting

Provides retail analytics including customer data integration,
demand forecasting, and recommendation engines.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import random


@dataclass
class Customer:
    """Represents a retail customer"""
    customer_id: str
    name: str
    email: str
    purchases: List[Dict] = field(default_factory=list)
    preferences: Dict = field(default_factory=dict)
    lifetime_value: float = 0.0


class CustomerDataIntegrator:
    """
    Integrate customer data from multiple sources.
    
    Uses swarm intelligence to merge and reconcile customer records,
    similar to how bees integrate information from multiple scouts.
    """
    
    def __init__(self):
        self.customers: Dict[str, Customer] = {}
        self.integration_count = 0
        
    def integrate_customer(
        self,
        customer_id: str,
        data_sources: List[Dict],
    ) -> bool:
        """
        Integrate customer data from multiple sources.
        
        Returns True if successful.
        """
        if not data_sources:
            return False
        
        # Merge data from sources
        merged_data = {}
        for source in data_sources:
            for key, value in source.items():
                if key not in merged_data:
                    merged_data[key] = value
                # Conflict resolution: prefer most recent
                elif "timestamp" in source and "timestamp" in merged_data.get(key, {}):
                    if source.get("timestamp", 0) > merged_data[key].get("timestamp", 0):
                        merged_data[key] = value
        
        # Create or update customer
        if customer_id not in self.customers:
            self.customers[customer_id] = Customer(
                customer_id=customer_id,
                name=merged_data.get("name", ""),
                email=merged_data.get("email", ""),
                purchases=merged_data.get("purchases", []),
                preferences=merged_data.get("preferences", {}),
            )
        else:
            customer = self.customers[customer_id]
            customer.purchases.extend(merged_data.get("purchases", []))
            customer.preferences.update(merged_data.get("preferences", {}))
        
        # Calculate lifetime value
        customer = self.customers[customer_id]
        customer.lifetime_value = sum(p.get("amount", 0) for p in customer.purchases)
        
        self.integration_count += 1
        return True
    
    def get_customer_360(self, customer_id: str) -> Optional[Dict]:
        """Get comprehensive customer view"""
        if customer_id not in self.customers:
            return None
        
        customer = self.customers[customer_id]
        
        return {
            "customer_id": customer.customer_id,
            "name": customer.name,
            "email": customer.email,
            "total_purchases": len(customer.purchases),
            "lifetime_value": customer.lifetime_value,
            "preferences": customer.preferences,
            "recent_purchases": customer.purchases[-5:],
        }
    
    def get_integration_stats(self) -> Dict:
        """Get integration statistics"""
        return {
            "total_customers": len(self.customers),
            "integrations_performed": self.integration_count,
            "total_lifetime_value": sum(c.lifetime_value for c in self.customers.values()),
        }


class DemandForecaster:
    """
    Forecast product demand using swarm intelligence.
    
    Uses bee-inspired collective decision making to predict
    future demand based on historical patterns.
    """
    
    def __init__(self):
        self.historical_data: Dict[str, List[Dict]] = {}
        self.forecasts: Dict[str, Dict] = {}
        
    def add_historical_data(
        self,
        product_id: str,
        date: str,
        quantity_sold: int,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Add historical sales data"""
        if product_id not in self.historical_data:
            self.historical_data[product_id] = []
        
        self.historical_data[product_id].append({
            "date": date,
            "quantity": quantity_sold,
            "metadata": metadata or {},
            "timestamp": time.time(),
        })
    
    def forecast_demand(
        self,
        product_id: str,
        periods_ahead: int = 7,
    ) -> Optional[Dict]:
        """
        Forecast demand for a product.
        
        Returns forecast or None if insufficient data.
        """
        if product_id not in self.historical_data:
            return None
        
        history = self.historical_data[product_id]
        
        if len(history) < 3:
            return None  # Need minimum data
        
        # Simple moving average forecast
        recent_sales = [h["quantity"] for h in history[-30:]]
        avg_daily = sum(recent_sales) / len(recent_sales)
        
        # Detect trend
        if len(recent_sales) >= 7:
            first_half = sum(recent_sales[:len(recent_sales)//2])
            second_half = sum(recent_sales[len(recent_sales)//2:])
            trend = (second_half - first_half) / first_half if first_half > 0 else 0
        else:
            trend = 0
        
        # Generate forecast
        forecasted_values = []
        for i in range(periods_ahead):
            # Apply trend
            value = avg_daily * (1 + trend * (i + 1) / periods_ahead)
            forecasted_values.append(max(0, int(value)))
        
        forecast = {
            "product_id": product_id,
            "forecast_horizon": periods_ahead,
            "forecasted_values": forecasted_values,
            "average_demand": int(avg_daily),
            "trend": trend,
            "confidence": 0.7,  # Simplified
            "generated_at": time.time(),
        }
        
        self.forecasts[product_id] = forecast
        
        return forecast
    
    def get_forecast_stats(self) -> Dict:
        """Get forecasting statistics"""
        return {
            "products_with_history": len(self.historical_data),
            "total_forecasts": len(self.forecasts),
        }


class RecommendationEngine:
    """
    Product recommendation engine using collaborative filtering.
    
    Uses swarm wisdom to identify patterns in customer behavior,
    like how bees communicate about good food sources.
    """
    
    def __init__(self):
        self.user_item_matrix: Dict[str, Set[str]] = {}  # user -> products
        self.item_similarity: Dict[str, Dict[str, float]] = {}
        self.recommendations_generated = 0
        
    def add_interaction(
        self,
        user_id: str,
        product_id: str,
    ) -> None:
        """Record user-product interaction"""
        if user_id not in self.user_item_matrix:
            self.user_item_matrix[user_id] = set()
        
        self.user_item_matrix[user_id].add(product_id)
    
    def calculate_similarity(
        self,
        product_a: str,
        product_b: str,
    ) -> float:
        """
        Calculate Jaccard similarity between products.
        
        Based on users who purchased both.
        """
        users_a = {u for u, items in self.user_item_matrix.items() if product_a in items}
        users_b = {u for u, items in self.user_item_matrix.items() if product_b in items}
        
        if not users_a or not users_b:
            return 0.0
        
        intersection = len(users_a & users_b)
        union = len(users_a | users_b)
        
        return intersection / union if union > 0 else 0.0
    
    def recommend(
        self,
        user_id: str,
        n: int = 5,
    ) -> List[str]:
        """
        Generate product recommendations for a user.
        
        Returns list of recommended product IDs.
        """
        if user_id not in self.user_item_matrix:
            return []
        
        user_products = self.user_item_matrix[user_id]
        
        # Score all products not yet purchased
        all_products = set()
        for products in self.user_item_matrix.values():
            all_products.update(products)
        
        candidate_products = all_products - user_products
        
        if not candidate_products:
            return []
        
        # Score candidates based on similarity to user's products
        scores = {}
        for candidate in candidate_products:
            score = 0.0
            for user_product in user_products:
                similarity = self.calculate_similarity(user_product, candidate)
                score += similarity
            
            scores[candidate] = score
        
        # Sort by score and return top N
        recommendations = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        self.recommendations_generated += 1
        
        return [product_id for product_id, _ in recommendations[:n]]
    
    def get_recommendation_stats(self) -> Dict:
        """Get recommendation engine statistics"""
        return {
            "total_users": len(self.user_item_matrix),
            "total_interactions": sum(len(items) for items in self.user_item_matrix.values()),
            "recommendations_generated": self.recommendations_generated,
        }
