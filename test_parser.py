import sys
sys.path.append('src')
from src.data.beatmap_parser import BeatmapParser

parser = BeatmapParser()
result = parser.parse_beatmap('testmap.osu')

print('Type:', type(result))
print('Length:', len(result) if result else 'None')
if result and len(result) > 0:
    print('First object:', result[0])
    print('Keys in first object:', list(result[0].keys()) if isinstance(result[0], dict) else 'Not a dict')
else:
    print('No results or empty')