# -*- mode: python ; coding: utf-8 -*-
import sys
sys.setrecursionlimit(5000)

block_cipher = None


a = Analysis(['cli.py'],
             pathex=['C:\\Users\\paragjai\\Desktop\\AI@Home\\Face Mask Detection in Realtime\\Packaging_Inference_Code_into_an_executable\\multi_threading_local_camera'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='cli',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
