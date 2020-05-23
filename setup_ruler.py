from notebook.services.config.manager import ConfigManager
cm = ConfigManager()
cm.update('notebook', {'ruler_column': [80]})
