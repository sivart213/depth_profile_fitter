# -*- coding: utf-8 -*-
"""
Insert module description/summary.

Provide any or all of the following:
1. extended summary
2. routine listings/functions/classes
3. see also
4. notes
5. references
6. examples

@author: j2cle
Created on Thu Jun 27 12:23:35 2024
"""

# %% Imports
import numpy as np
import pandas as pd

# %% Code


import ctypes as ct
from ctypes import wintypes as w
from enum import IntFlag

class FSFlags(IntFlag):
    FILE_CASE_SENSITIVE_SEARCH        = 0x00000001
    FILE_CASE_PRESERVED_NAMES         = 0x00000002
    FILE_UNICODE_ON_DISK              = 0x00000004
    FILE_PERSISTENT_ACLS              = 0x00000008
    FILE_FILE_COMPRESSION             = 0x00000010
    FILE_VOLUME_QUOTAS                = 0x00000020
    FILE_SUPPORTS_SPARSE_FILES        = 0x00000040
    FILE_SUPPORTS_REPARSE_POINTS      = 0x00000080
    FILE_VOLUME_IS_COMPRESSED         = 0x00008000
    FILE_SUPPORTS_OBJECT_IDS          = 0x00010000
    FILE_SUPPORTS_ENCRYPTION          = 0x00020000
    FILE_NAMED_STREAMS                = 0x00040000
    FILE_READ_ONLY_VOLUME             = 0x00080000
    FILE_SEQUENTIAL_WRITE_ONCE        = 0x00100000
    FILE_SUPPORTS_TRANSACTIONS        = 0x00200000
    FILE_SUPPORTS_HARD_LINKS          = 0x00400000
    FILE_SUPPORTS_EXTENDED_ATTRIBUTES = 0x00800000
    FILE_SUPPORTS_OPEN_BY_FILE_ID     = 0x01000000
    FILE_SUPPORTS_USN_JOURNAL         = 0x02000000
    FILE_SUPPORTS_BLOCK_REFCOUNTING   = 0x08000000
    FILE_DAX_VOLUME                   = 0x20000000

# def validate(result,func,args):
#     if not result:
#         raise ct.WinError(ct.get_last_error())
#     return None

dll = ct.WinDLL('kernel32',use_last_error=True)
dll.GetVolumeInformationW.argtypes = w.LPCWSTR,w.LPWSTR,w.DWORD,w.LPDWORD,w.LPDWORD,w.LPDWORD,w.LPWSTR,w.DWORD
dll.GetVolumeInformationW.restype = w.BOOL
# dll.GetVolumeInformationW.errcheck = validate

volumeNameBuffer = ct.create_unicode_buffer(w.MAX_PATH + 1)
fileSystemNameBuffer = ct.create_unicode_buffer(w.MAX_PATH + 1)
serial_number = w.DWORD()
max_component_length = w.DWORD() 
file_system_flags = w.DWORD()

target_disk = 'z:\\'

dll.GetVolumeInformationW(target_disk,
                          volumeNameBuffer, ct.sizeof(volumeNameBuffer),
                          ct.byref(serial_number),
                          ct.byref(max_component_length),
                          ct.byref(file_system_flags),
                          fileSystemNameBuffer, ct.sizeof(fileSystemNameBuffer))

mount_point = target_disk[:-1]
disk_label = volumeNameBuffer.value
fs_type = fileSystemNameBuffer.value
max_length = max_component_length.value
flags = FSFlags(file_system_flags.value)
serial = serial_number.value

print(f'{mount_point=}\n{disk_label=}\n{fs_type=}\n{max_length=}\n{flags=}\n{serial=}')

# dll = ct.WinDLL('kernel32',use_last_error=True)
# dll.GetVolumeInformationW.argtypes = w.LPCWSTR,w.LPDWORD
# dll.GetVolumeInformationW.restype = w.BOOL
# # dll.GetVolumeInformationW.errcheck = validate

# volumeNameBuffer = ct.create_unicode_buffer(w.MAX_PATH + 1)
# fileSystemNameBuffer = ct.create_unicode_buffer(w.MAX_PATH + 1)
# serial_number = w.DWORD()
# max_component_length = w.DWORD() 
# file_system_flags = w.DWORD()

# target_disk = 'C:\\'

# dll.GetVolumeInformationW(target_disk,
#                           ct.byref(serial_number))



# mount_point = target_disk[:-1]
# disk_label = volumeNameBuffer.value
# fs_type = fileSystemNameBuffer.value
# max_length = max_component_length.value
# flags = FSFlags(file_system_flags.value)
# serial = serial_number.value

# print(f'{mount_point=}\n{serial=}')