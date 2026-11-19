// ProxiChorizo License 1.0; see LICENSE.
// Exact first-two-round partial evaluation for word 14 plus word 15.
// ParamsCore (107 words) is followed by Plan01 snapshots and pre-additions.
// Generated from the same Blake2b dependency graph as txid::Plan; no rounds skipped.
#define TX_PLAN29 { \
v[0]=P[P_VI+0]; \
v[1]=P[P_VI+1]; \
v[2]=P[P_VI+2]; \
v[3]=P[P_VI+3]; \
v[4]=P[P_VI+4]; \
v[5]=P[P_VI+5]; \
v[6]=P[P_VI+6]; \
v[7]=P[P_VI+7]; \
v[8]=P[P_VI+8]; \
v[9]=P[P_VI+9]; \
v[10]=P[P_VI+10]; \
v[11]=P[P_VI+11]; \
v[12]=P[P_VI+12]; \
v[13]=P[P_VI+13]; \
v[14]=P[P_VI+14]; \
v[15]=P[P_VI+15]; \
v[0]=P[107+0]; \
v[12]=P[107+1]; \
v[8]=P[107+2]; \
v[4]=P[107+3]; \
v[0]=P[107+4]; \
v[12]=P[107+5]; \
v[8]=P[107+6]; \
v[4]=P[107+7]; \
v[1]=P[107+8]; \
v[13]=P[107+9]; \
v[9]=P[107+10]; \
v[5]=P[107+11]; \
v[1]=P[107+12]; \
v[13]=P[107+13]; \
v[9]=P[107+14]; \
v[5]=P[107+15]; \
v[2]=P[107+16]; \
v[14]=P[107+17]; \
v[10]=P[107+18]; \
v[6]=P[107+19]; \
v[2]=P[107+20]; \
v[14]=P[107+21]; \
v[10]=P[107+22]; \
v[6]=P[107+23]; \
v[3]=P[107+24]; \
v[15]=P[107+25]; \
v[11]=P[107+26]; \
v[7]=P[107+27]; \
v[3]=P[107+28]; \
v[15]=P[107+29]; \
v[11]=P[107+30]; \
v[7]=P[107+31]; \
v[0]=P[107+32]; \
v[15]=P[107+33]; \
v[10]=P[107+34]; \
v[5]=P[107+35]; \
v[0]=P[107+36]; \
v[15]=P[107+37]; \
v[10]=P[107+38]; \
v[5]=P[107+39]; \
v[1]=P[107+40]; \
v[12]=P[107+41]; \
v[11]=P[107+42]; \
v[6]=P[107+43]; \
v[1]=P[107+44]; \
v[12]=P[107+45]; \
v[11]=P[107+46]; \
v[6]=P[107+47]; \
v[2]=P[107+48]; \
v[13]=P[107+49]; \
v[8]=P[107+50]; \
v[7]=P[107+51]; \
v[2]=P[107+52]; \
v[13]=P[107+53]; \
v[8]=P[107+54]; \
v[7]=P[107+55]; \
v[3]=m[14]+P[235+14]; \
v[14]=TX_ROR(v[14]^v[3],32); \
v[9]=v[9]+v[14]; \
v[4]=TX_ROR(v[4]^v[9],24); \
v[3]=v[3]+v[4]+m[15]; \
v[14]=TX_ROR(v[14]^v[3],16); \
v[9]=v[9]+v[14]; \
v[4]=TX_ROR(v[4]^v[9],63); \
v[0]=v[0]+v[4]+m[14]; \
v[12]=TX_ROR(v[12]^v[0],32); \
v[8]=v[8]+v[12]; \
v[4]=TX_ROR(v[4]^v[8],24); \
v[0]=v[0]+v[4]+m[10]; \
v[12]=TX_ROR(v[12]^v[0],16); \
v[8]=v[8]+v[12]; \
v[4]=TX_ROR(v[4]^v[8],63); \
v[1]=P[107+72]; \
v[13]=P[107+73]; \
v[9]=v[9]+v[13]; \
v[5]=TX_ROR(v[5]^v[9],24); \
v[1]=v[5]+P[235+19]; \
v[13]=TX_ROR(v[13]^v[1],16); \
v[9]=v[9]+v[13]; \
v[5]=TX_ROR(v[5]^v[9],63); \
v[2]=P[107+80]; \
v[14]=TX_ROR(v[14]^v[2],32); \
v[10]=v[10]+v[14]; \
v[6]=TX_ROR(v[6]^v[10],24); \
v[2]=v[2]+v[6]+m[15]; \
v[14]=TX_ROR(v[14]^v[2],16); \
v[10]=v[10]+v[14]; \
v[6]=TX_ROR(v[6]^v[10],63); \
v[3]=v[3]+P[235+22]; \
v[15]=TX_ROR(v[15]^v[3],32); \
v[11]=v[11]+v[15]; \
v[7]=TX_ROR(v[7]^v[11],24); \
v[3]=v[3]+v[7]+m[6]; \
v[15]=TX_ROR(v[15]^v[3],16); \
v[11]=v[11]+v[15]; \
v[7]=TX_ROR(v[7]^v[11],63); \
v[0]=v[0]+v[5]+m[1]; \
v[15]=TX_ROR(v[15]^v[0],32); \
v[10]=v[10]+v[15]; \
v[5]=TX_ROR(v[5]^v[10],24); \
v[0]=v[0]+v[5]+m[12]; \
v[15]=TX_ROR(v[15]^v[0],16); \
v[10]=v[10]+v[15]; \
v[5]=TX_ROR(v[5]^v[10],63); \
v[1]=v[1]+v[6]+m[0]; \
v[12]=TX_ROR(v[12]^v[1],32); \
v[11]=v[11]+v[12]; \
v[6]=TX_ROR(v[6]^v[11],24); \
v[1]=v[1]+v[6]+m[2]; \
v[12]=TX_ROR(v[12]^v[1],16); \
v[11]=v[11]+v[12]; \
v[6]=TX_ROR(v[6]^v[11],63); \
v[2]=v[2]+v[7]+m[11]; \
v[13]=TX_ROR(v[13]^v[2],32); \
v[8]=v[8]+v[13]; \
v[7]=TX_ROR(v[7]^v[8],24); \
v[2]=v[2]+v[7]+m[7]; \
v[13]=TX_ROR(v[13]^v[2],16); \
v[8]=v[8]+v[13]; \
v[7]=TX_ROR(v[7]^v[8],63); \
v[3]=v[3]+v[4]+m[5]; \
v[14]=TX_ROR(v[14]^v[3],32); \
v[9]=v[9]+v[14]; \
v[4]=TX_ROR(v[4]^v[9],24); \
v[3]=v[3]+v[4]+m[3]; \
v[14]=TX_ROR(v[14]^v[3],16); \
v[9]=v[9]+v[14]; \
v[4]=TX_ROR(v[4]^v[9],63); \
}

__kernel __attribute__((reqd_work_group_size(TX_WG,1,1)))
void txid_plan29(__constant ulong *P, ulong nbase, ulong count, uint iters,
                 __global ulong2 *hits, __global uint *nhit) {
    const uint gid = (uint)get_global_id(0), stride = (uint)get_global_size(0);
    uint n = (uint)nbase + gid;
    const uint limit = (uint)(count / stride) + (gid < (uint)(count % stride));
    // The host splits launches at low-word wrap; high nonce bits stay fixed.
    const ulong high = nbase & 0xffffffff00000000UL;
    for (uint i = 0; i < limit; ++i, n += stride) {
        const ulong physical = high | (ulong)n, X = tx_bswap(physical);
        ulong m[16], v[16];
        _Pragma("unroll") for (int j = 0; j < 14; ++j) m[j] = P[j];
        const uint sh = (uint)P[P_SHIFT] & 63u;
        m[14] = P[14] | (X << sh); m[15] = X >> (64u - sh);
        TX_PLAN29 TX_ROUNDS(2)
        const ulong h3 = P[P_H+3] ^ v[3] ^ v[11];
        if ((h3 & P[P_MASK]) == 0) {
            uint at = atomic_inc(nhit);
            if (at < TX_MAX_HITS) hits[at] = (ulong2)(physical, h3);
        }
    }
}
