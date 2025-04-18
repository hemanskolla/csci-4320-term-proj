#ifndef CLOCKCYCLE_H
#define CLOCKCYCLE_H
#include <stdint.h>

uint64_t clock_now(void)
{
    unsigned int tbl, tbu0, tbul;
    do
    {
        __asm__ __volatile__("mftbu %0" : "=r"(tbu0));
        __asm__ __volatile__("mftb %0" : "=r"(tbl));
        __asm__ __volatile__("mftbu %0" : "=r"(tbul));
    } while (tbu0 != tbul);
    return (((uint64_t)tbu0) << 32) | tbl;
}
#endif