
import sys
import os
import shutil
import json
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from api.routes.chapter import twopass

def test_status_persistence():
    # Setup
    chapter_id = "test_persistence_ch001"
    upload_dir = Path("./uploads/chapters") / chapter_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    status_data = {
        "chapter_id": chapter_id,
        "status": "generating",
        "progress": 45,
        "message": "Testing persistence"
    }
    
    try:
        # Test Save
        twopass.save_chapter_status_to_disk(chapter_id, status_data)
        
        status_file = upload_dir / "status.json"
        assert status_file.exists()
        
        with open(status_file, 'r') as f:
            saved = json.load(f)
            assert saved['progress'] == 45
            assert saved['status'] == "generating"

        # Test Load
        loaded = twopass.load_chapter_status_from_disk(chapter_id)
        assert loaded is not None
        assert loaded['message'] == "Testing persistence"
        
        # Test Memory Rehydration in get_chapter_status
        # Mock chapter_status to be empty
        with patch.dict(twopass.chapter_status, {}, clear=True):
            # We need to run the async function
            import asyncio
            
            # Since get_chapter_status is async, we need a loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Mock ChapterDatabase to NOT exist so it falls through to persistence check
            with patch('api.routes.chapter.twopass.ChapterDatabase') as MockDB:
                mock_db_instance = MagicMock()
                mock_db_instance.exists.return_value = False 
                # Note: The logic in get_chapter_status checks memory first, then load_from_disk
                # If not in memory, it calls load_chapter_status_from_disk
                # My implementation:
                # if chapter_id not in chapter_status:
                #    disk = load_from_disk()
                #    if disk: ... 
                #    else: check_db()
                
                response = loop.run_until_complete(twopass.get_chapter_status(chapter_id))
                
                assert response.status == "generating"
                assert response.progress == 45
                
                # Verify it was added back to memory
                assert chapter_id in twopass.chapter_status
            
            loop.close()

    finally:
        # Cleanup
        if upload_dir.exists():
            shutil.rmtree(upload_dir)

if __name__ == "__main__":
    test_status_persistence()
    print("Test passed!")
