#!/usr/bin/env python3
"""
YouTube video downloader for collecting panic detection training data.

Usage:
    # Single video
    python scripts/download_youtube_videos.py \
        --url "https://www.youtube.com/watch?v=VIDEO_ID" \
        --output data/panic_videos
    
    # Multiple videos from file
    python scripts/download_youtube_videos.py \
        --file urls.txt \
        --output data/normal_videos \
        --label normal
    
    # Playlist
    python scripts/download_youtube_videos.py \
        --url "https://www.youtube.com/playlist?list=PLAYLIST_ID" \
        --output data/panic_videos \
        --label panic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

try:
    import yt_dlp
except ImportError:
    print("ERROR: yt-dlp not installed")
    print("Install with: pip install yt-dlp")
    sys.exit(1)


class YouTubeDownloader:
    """Download YouTube videos with yt-dlp."""
    
    def __init__(
        self,
        output_dir: str | Path,
        quality: str = "720p",
        format: str = "mp4",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quality = quality
        self.format = format
        
        # Map quality to format selector
        quality_map = {
            "360p": "bestvideo[height<=360]+bestaudio/best[height<=360]",
            "480p": "bestvideo[height<=480]+bestaudio/best[height<=480]",
            "720p": "bestvideo[height<=720]+bestaudio/best[height<=720]",
            "1080p": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
            "best": "bestvideo+bestaudio/best",
        }
        
        self.format_selector = quality_map.get(quality, quality_map["720p"])
    
    def download(
        self,
        url: str,
        label: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> list[str]:
        """
        Download video(s) from URL.
        
        Args:
            url: YouTube video or playlist URL
            label: Optional label prefix for filename
            start_time: Optional start time (format: HH:MM:SS or seconds)
            end_time: Optional end time (format: HH:MM:SS or seconds)
            
        Returns:
            List of downloaded file paths
        """
        # Build output template
        if label:
            output_template = str(self.output_dir / f"{label}_%(title)s_%(id)s.%(ext)s")
        else:
            output_template = str(self.output_dir / "%(title)s_%(id)s.%(ext)s")
        
        # yt-dlp options
        ydl_opts = {
            "format": self.format_selector,
            "outtmpl": output_template,
            "merge_output_format": self.format,
            "quiet": False,
            "no_warnings": False,
            "progress_hooks": [self._progress_hook],
        }
        
        # Add time trimming if specified
        postprocessor_args = []
        if start_time or end_time:
            if start_time:
                postprocessor_args.extend(["-ss", start_time])
            if end_time:
                postprocessor_args.extend(["-to", end_time])
            
            ydl_opts["postprocessors"] = [{
                "key": "FFmpegVideoConvertor",
                "preferedformat": self.format,
            }]
            ydl_opts["postprocessor_args"] = postprocessor_args
        
        downloaded_files = []
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Get info first
                info = ydl.extract_info(url, download=False)
                
                if info is None:
                    print(f"ERROR: Could not extract info from {url}")
                    return downloaded_files
                
                # Check if it's a playlist
                if "entries" in info:
                    print(f"\nPlaylist detected: {info.get('title', 'Unknown')}")
                    print(f"Total videos: {len(info['entries'])}")
                    
                    for idx, entry in enumerate(info["entries"], 1):
                        if entry is None:
                            continue
                        print(f"\n[{idx}/{len(info['entries'])}] Downloading: {entry.get('title', 'Unknown')}")
                        try:
                            result = ydl.extract_info(entry["url"], download=True)
                            if result:
                                filename = ydl.prepare_filename(result)
                                downloaded_files.append(filename)
                                print(f"✓ Saved: {filename}")
                        except Exception as e:
                            print(f"✗ Error downloading video {idx}: {e}")
                            continue
                else:
                    # Single video
                    print(f"\nDownloading: {info.get('title', 'Unknown')}")
                    print(f"Duration: {info.get('duration', 0) // 60}:{info.get('duration', 0) % 60:02d}")
                    
                    result = ydl.extract_info(url, download=True)
                    if result:
                        filename = ydl.prepare_filename(result)
                        downloaded_files.append(filename)
                        print(f"✓ Saved: {filename}")
        
        except Exception as e:
            print(f"ERROR: {e}")
            return downloaded_files
        
        return downloaded_files
    
    @staticmethod
    def _progress_hook(d):
        """Progress hook for yt-dlp."""
        if d["status"] == "downloading":
            percent = d.get("_percent_str", "N/A")
            speed = d.get("_speed_str", "N/A")
            eta = d.get("_eta_str", "N/A")
            print(f"\rDownloading: {percent} at {speed} ETA: {eta}", end="", flush=True)
        elif d["status"] == "finished":
            print(f"\rDownload complete, processing...                    ")


def load_urls_from_file(filepath: str | Path) -> list[str]:
    """Load URLs from a text file (one URL per line)."""
    urls = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    return urls


def main():
    parser = argparse.ArgumentParser(
        description="Download YouTube videos for panic detection training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download single video
  python scripts/download_youtube_videos.py \\
      --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \\
      --output data/videos

  # Download from URL list
  python scripts/download_youtube_videos.py \\
      --file urls.txt \\
      --output data/panic_videos \\
      --label panic

  # Download playlist
  python scripts/download_youtube_videos.py \\
      --url "https://www.youtube.com/playlist?list=PLxxx" \\
      --output data/normal_videos \\
      --label normal \\
      --quality 720p

  # Download with time trimming
  python scripts/download_youtube_videos.py \\
      --url "https://www.youtube.com/watch?v=VIDEO_ID" \\
      --output data/videos \\
      --start-time 00:01:30 \\
      --end-time 00:05:00
        """
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--url",
        help="YouTube video or playlist URL",
    )
    input_group.add_argument(
        "--file",
        help="Text file with URLs (one per line)",
    )
    
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for downloaded videos",
    )
    parser.add_argument(
        "--label",
        help="Label prefix for filenames (e.g., 'panic', 'normal')",
    )
    parser.add_argument(
        "--quality",
        default="720p",
        choices=["360p", "480p", "720p", "1080p", "best"],
        help="Video quality (default: 720p)",
    )
    parser.add_argument(
        "--format",
        default="mp4",
        help="Output video format (default: mp4)",
    )
    parser.add_argument(
        "--start-time",
        help="Start time for trimming (format: HH:MM:SS or seconds)",
    )
    parser.add_argument(
        "--end-time",
        help="End time for trimming (format: HH:MM:SS or seconds)",
    )
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = YouTubeDownloader(
        output_dir=args.output,
        quality=args.quality,
        format=args.format,
    )
    
    # Get URLs
    urls = []
    if args.url:
        urls = [args.url]
    elif args.file:
        print(f"Loading URLs from: {args.file}")
        urls = load_urls_from_file(args.file)
        print(f"Found {len(urls)} URL(s)")
    
    if not urls:
        print("ERROR: No URLs to download")
        return 1
    
    # Download videos
    all_downloaded = []
    for idx, url in enumerate(urls, 1):
        print(f"\n{'='*60}")
        print(f"Processing URL {idx}/{len(urls)}")
        print(f"{'='*60}")
        
        downloaded = downloader.download(
            url=url,
            label=args.label,
            start_time=args.start_time,
            end_time=args.end_time,
        )
        all_downloaded.extend(downloaded)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"Total videos downloaded: {len(all_downloaded)}")
    print(f"Output directory: {args.output}")
    
    if all_downloaded:
        print(f"\nDownloaded files:")
        for filepath in all_downloaded:
            print(f"  - {filepath}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
