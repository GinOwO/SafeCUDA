/**
* @file memtable.h
 * @brief SafeCUDA metadata header file
 *
 * This file contains the header file for the metadata table
 *
 * @author Anirudh <anirudh.sridhar2022@vitstudent.ac.in>
 * @date 2025-09-22
 * @version 0.0.1
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 *
 * Change Log:
 * - 2025-09-22: Initial implementation
 * - 2025-10-22: Removed redundancies and made padding work based on arch
 */
#ifndef MEMTABLE_H
#define MEMTABLE_H

#include "safecache.cuh"

#include <cstdint>

namespace safecuda::memtable
{

constexpr std::int16_t MAGIC_WORD = 0x5AFE;

struct Header {
	std::int16_t magic_word = MAGIC_WORD;
	std::int8_t _pad1[6] = {0};
	cache::CacheEntry *entry = nullptr;
	std::uintptr_t *memory = nullptr;
};

std::uintptr_t *init_header(void *base_ptr);
bool validate_header(void *base_ptr);
void delete_entry(void *base_ptr);

}

#endif // !MEMTABLE_H
