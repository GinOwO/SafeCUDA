
#ifndef MEMTABLE_H
#define MEMTABLE_H

#include <cstdint>

namespace safecuda::memtable
{

constexpr std::int16_t MAGIC_WORD = 0x5AFE;

struct Entry {
	std::uint32_t start_addr;
	std::uint32_t block_size;
	std::uint8_t flags;
	std::uint32_t epochs;
};

struct Header {
	std::int16_t magic_word = MAGIC_WORD;
	std::int8_t _pad1[6];
	Entry *entry;
	std::uint32_t *memory;
};

void init_entry();
bool validate_header();

}

#endif // !MEMTABLE_H
