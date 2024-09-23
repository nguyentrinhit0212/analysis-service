from app.app_factory import AppFactory

# Create an instance of AppFactory
app_factory = AppFactory()

# Create the Flask application using the factory
app = app_factory.create_app()

# Expose the app instance for WSGI servers
__all__ = ['app']