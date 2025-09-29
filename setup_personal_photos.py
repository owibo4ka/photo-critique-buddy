"""
Script to help set up your personal photo collection for the Photo Critique Buddy
"""

import os
import shutil
from pathlib import Path

def setup_directories():
    """Create the necessary directories for personal photos"""
    # Create directories outside the repo for privacy
    home_dir = Path.home()
    photo_base_dir = home_dir / "pcb-data"
    
    directories = [
        photo_base_dir / "personal" / "strong",
        photo_base_dir / "personal" / "maybe", 
        photo_base_dir / "personal" / "weak",
        photo_base_dir / "models",
        "uploads"  # Keep uploads in repo for temporary storage
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def copy_sample_photos():
    """Copy some sample photos to get started"""
    print("\n📸 Setting up sample photos...")
    print("You can replace these with your own photos later.")
    
    # Create some placeholder files to show the structure
    home_dir = Path.home()
    photo_base_dir = home_dir / "pcb-data"
    
    sample_dirs = {
        photo_base_dir / "personal" / "strong": "Put your best photos here (portfolio-worthy)",
        photo_base_dir / "personal" / "maybe": "Put uncertain photos here (decent but not great)",
        photo_base_dir / "personal" / "weak": "Put photos you wouldn't show here (learning examples)"
    }
    
    for directory, description in sample_dirs.items():
        readme_path = Path(directory) / "README.txt"
        with open(readme_path, "w") as f:
            f.write(f"{description}\n\n")
            f.write("Supported formats: .jpg, .jpeg, .png, .gif, .bmp, .tiff, .webp\n")
            f.write("Recommended: Start with 20-50 photos in each category\n")
        print(f"✓ Created {readme_path}")

def print_instructions():
    """Print setup instructions"""
    print("\n" + "="*60)
    print("🎯 PHOTO CRITIQUE BUDDY SETUP")
    print("="*60)
    home_dir = Path.home()
    photo_base_dir = home_dir / "pcb-data"
    
    print("\n1. 📁 Directory Structure Created:")
    print(f"   {photo_base_dir}/personal/strong/  - Your best photos")
    print(f"   {photo_base_dir}/personal/maybe/   - Decent photos") 
    print(f"   {photo_base_dir}/personal/weak/    - Photos to learn from")
    print(f"   {photo_base_dir}/models/           - AI models will be stored here")
    print("   uploads/                            - Uploaded photos for analysis")
    
    print("\n2. 📸 Next Steps:")
    print(f"   • Copy 20-50 of your BEST photos to {photo_base_dir}/personal/strong/")
    print(f"   • Copy 20-50 photos you're unsure about to {photo_base_dir}/personal/maybe/")
    print(f"   • Copy 20-50 photos you wouldn't show to {photo_base_dir}/personal/weak/")
    print("   • The more photos you add, the better the personalization!")
    
    print("\n3. 🚀 Then run your FastAPI server:")
    print("   python -m app.main")
    print("   Visit: http://localhost:8001/docs")
    
    print("\n4. 🧠 The AI will learn your taste from these photos and:")
    print("   • Score new photos based on your preferences")
    print("   • Find similar photos from your collection")
    print("   • Give personalized critique and suggestions")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    print("Setting up Photo Critique Buddy...")
    setup_directories()
    copy_sample_photos()
    print_instructions()
