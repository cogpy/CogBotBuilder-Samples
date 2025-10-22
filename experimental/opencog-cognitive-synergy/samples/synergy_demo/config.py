"""Configuration for Synergy Demonstration"""

import os


class DefaultConfig:
    """Synergy Demo Configuration"""
    
    PORT = 3979  # Different port from basic sample
    APP_ID = os.environ.get("MicrosoftAppId", "")
    APP_PASSWORD = os.environ.get("MicrosoftAppPassword", "")
    APP_TYPE = os.environ.get("MicrosoftAppType", "MultiTenant")
    APP_TENANTID = os.environ.get("MicrosoftAppTenantId", "")