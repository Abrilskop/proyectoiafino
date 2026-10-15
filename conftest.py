import sys
from pathlib import Path

# Hace importables los paquetes src/, deployment/ y monitoring/ durante pytest.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
