# Trace format (version 1)
Little-endian binary. 64-byte header, then fixed 24-byte records until EOF.

Header: `magic "ACTRACE\0"` (8) | `version=1` u32 | `rec_size=24` u32 | `num_records` u64 | `total_icount` u64 | 32 reserved.
`num_records` and `total_icount` are patched when the output is a regular file. On a pipe they stay 0 (unknown); readers read until EOF.

Record: `addr` u64 | `pc` u64 | `icount` u32 | `size` u8 | `flags` u8 (bit0 = store) | `pad` u16.

| Field | Raw trace (lackey2trace) | LLC trace (cachesim filter) |
|---|---|---|
| addr | byte address (virtual) | line-aligned address |
| pc | address of the last instruction line seen before the access | pc of the access that missed L2 |
| icount | instructions since the previous record, including this one | instructions since the previous LLC record (instructions of L1/L2 hits are added) |
| size | access size in bytes | line size |

Raw trace covers only the ROI (between the first and second store to `roi_marker`).
`total_icount`: raw = ROI instructions; LLC trace = all raw instructions. Instructions after the last LLC access are not in any record.
