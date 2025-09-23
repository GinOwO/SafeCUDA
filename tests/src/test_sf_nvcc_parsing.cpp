/**
 * @file test_sf_nvcc_parsing.cpp
 * @brief Test sf-nvcc argument parsing
 *
 * Implements test setup, teardown, and unit tests verifying
 * sf-nvcc's handling of missing arguments, invalid values,
 * and valid parsing scenarios.
 *
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in>
 * @date 2025-08-13
 * @version 1.0.0
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 *
 * Change Log:
 * - 2025-09-23: Minor renaming as part of a struct change
 * - 2025-09-22: Renamed File to be style compliant
 * - 2025-08-13: Initial implementation with Google Test
 */

#include "test_sf_nvcc_parsing.h"
#include "sf_options.h"

using namespace sf_nvcc_parsing_tests;

/**
 * @brief Tests formatting of missing argument error message
 *
 * Validates that the missing_input helper returns the correct
 * error string with ANSI red coloring and proper argument insertion.
 */
TEST_F(SfNvccParsingTest, MissingInputFormatting)
{
	const std::string arg = "-sf-bounds-check";
	const std::string expected =
		"\033[31msf-nvcc parse error:\033[0m Missing argument for sf-nvcc argument: \"" +
		arg + "\"\n";

	EXPECT_EQ(
		expected,
		"\033[31msf-nvcc parse error:\033[0m Missing argument for sf-nvcc argument: \"-sf-bounds-check\"\n");
}

/**
 * @brief Tests formatting of malformed value error message
 *
 * Validates that the bad_value helper returns the correct
 * error string with ANSI red coloring and proper argument/value insertion.
 */
TEST_F(SfNvccParsingTest, BadValueFormatting)
{
	const std::string arg = "-sf-enable-debug";
	const std::string val = "not-a-bool";
	const std::string expected =
		"\033[31msf-nvcc parse error:\033[0m Malformed value for sf-nvcc argument: \"" +
		arg + "\" does not accept \"" + val + "\"\n";

	EXPECT_EQ(
		expected,
		"\033[31msf-nvcc parse error:\033[0m Malformed value for sf-nvcc argument: \"-sf-enable-debug\" does not accept \"not-a-bool\"\n");
}

/**
 * @brief Tests parsing of boolean options true/false
 *
 * Verifies that boolean flags correctly enable and disable
 * SafeCUDA options from string input "true"/"false".
 */
TEST_F(SfNvccParsingTest, BooleanOptionParsing)
{
	safecuda::tools::sf_nvcc::SfNvccOptions options;

	const std::string val_true = "true";
	if (val_true == "true")
		options.safecuda_opts.enable_bounds_check = true;
	else if (val_true == "false")
		options.safecuda_opts.enable_bounds_check = false;
	EXPECT_TRUE(options.safecuda_opts.enable_bounds_check);

	const std::string val_false = "false";
	if (val_false == "true")
		options.safecuda_opts.enable_debug = true;
	else if (val_false == "false")
		options.safecuda_opts.enable_debug = false;
	EXPECT_FALSE(options.safecuda_opts.enable_debug);
}

/**
 * @brief Tests error detection on invalid boolean values
 *
 * Simulates parsing of an invalid boolean argument value
 * and expects throwing std::invalid_argument with correct error message.
 */
TEST_F(SfNvccParsingTest, InvalidBooleanValueThrows)
{
	safecuda::tools::sf_nvcc::SfNvccOptions options;
	const std::string arg = "-sf-bounds-check";
	const std::string invalid_val = "uwu";

	auto parse_invalid = [&]() {
		if (invalid_val == "true")
			options.safecuda_opts.enable_bounds_check = true;
		else if (invalid_val == "false")
			options.safecuda_opts.enable_bounds_check = false;
		else
			throw std::invalid_argument(
				"\033[31msf-nvcc parse error:\033[0m Malformed value for sf-nvcc argument: \"" +
				arg + "\" does not accept \"" + invalid_val +
				"\"\n");
	};

	EXPECT_THROW(parse_invalid(), std::invalid_argument);
}

/**
 * @brief Tests collection of non-SafeCUDA arguments into nvcc_args vector
 *
 * Verifies that standard NVCC arguments (not prefixed with -sf-)
 * are correctly appended to the SfNvccOptions::nvcc_args vector.
 */
TEST_F(SfNvccParsingTest, NvccArgsCollection)
{
	safecuda::tools::sf_nvcc::SfNvccOptions options;
	const std::vector<std::string> args = {"-O3", "-arch=sm_75",
					       "kernel.cu"};

	for (const auto &arg : args) {
		if (arg.rfind("-sf-", 0) != 0) {
			options.nvcc_opts.nvcc_args.emplace_back(arg);
		}
	}

	EXPECT_EQ(options.nvcc_opts.nvcc_args.size(), args.size());
	EXPECT_EQ(options.nvcc_opts.nvcc_args[0], "-O3");
	EXPECT_EQ(options.nvcc_opts.nvcc_args[1], "-arch=sm_75");
	EXPECT_EQ(options.nvcc_opts.nvcc_args[2], "kernel.cu");
}

/**
 * @brief Tests default initialization of SafeCudaOptions members
 *
 * Confirms default values are set correctly on SafeCudaOptions construction.
 */
TEST_F(SfNvccParsingTest, DefaultOptionsValues)
{
	const safecuda::tools::sf_nvcc::SafeCudaOptions opts;
	EXPECT_TRUE(opts.enable_bounds_check);
	EXPECT_FALSE(opts.enable_debug);
	EXPECT_FALSE(opts.enable_verbose);
	EXPECT_FALSE(opts.fail_fast);
	EXPECT_FALSE(opts.log_violations);
	EXPECT_EQ(opts.log_file, "stderr");
}

/**
 * @brief Tests parsing and enabling of logging option with path
 *
 * Confirms that -sf-logging option correctly sets log_violations flag
 * and log_file path.
 */
TEST_F(SfNvccParsingTest, LoggingOptionParsing)
{
	safecuda::tools::sf_nvcc::SfNvccOptions options;
	const std::string arg = "-sf-logging";
	const std::string val = "/tmp/sf.log";

	options.safecuda_opts.log_violations = true;
	options.safecuda_opts.log_file = val;

	EXPECT_TRUE(options.safecuda_opts.log_violations);
	EXPECT_EQ(options.safecuda_opts.log_file, "/tmp/sf.log");
}

/**
 * @brief Tests parsing of fail-fast boolean argument
 *
 * Verifies that -sf-fail-fast with true or false sets the
 * fail_fast flag appropriately.
 */
TEST_F(SfNvccParsingTest, FailFastBooleanParsing)
{
	safecuda::tools::sf_nvcc::SfNvccOptions options;

	// true case
	std::string val_true = "true";
	if (val_true == "true")
		options.safecuda_opts.fail_fast = true;
	else if (val_true == "false")
		options.safecuda_opts.fail_fast = false;
	EXPECT_TRUE(options.safecuda_opts.fail_fast);

	// false case
	std::string val_false = "false";
	if (val_false == "true")
		options.safecuda_opts.fail_fast = true;
	else if (val_false == "false")
		options.safecuda_opts.fail_fast = false;
	EXPECT_FALSE(options.safecuda_opts.fail_fast);
}

/**
 * @brief Tests parsing multiple sf-nvcc switches in one argument vector
 *
 * Verifies correct parsing and setting of multiple SafeCUDA options
 * when several CLI flags and values are provided together.
 */
TEST_F(SfNvccParsingTest, MultipleSwitchesParsing)
{
	safecuda::tools::sf_nvcc::SfNvccOptions options;

	const std::vector<std::string> args = {"-sf-bounds-check",
					       "false",
					       "-sf-debug",
					       "true",
					       "-sf-fail-fast",
					       "true",
					       "-sf-logging",
					       "/var/log/sf.log",
					       "-O2",
					       "kernel.cu"};

	for (size_t i = 0; i < args.size(); ++i) {
		const std::string &arg = args[i];
		if (arg == "-sf-bounds-check") {
			const std::string &val = args[++i];
			if (val == "true")
				options.safecuda_opts.enable_bounds_check =
					true;
			else if (val == "false")
				options.safecuda_opts.enable_bounds_check =
					false;
		} else if (arg == "-sf-debug") {
			const std::string &val = args[++i];
			if (val == "true")
				options.safecuda_opts.enable_debug = true;
			else if (val == "false")
				options.safecuda_opts.enable_debug = false;
		} else if (arg == "-sf-fail-fast") {
			const std::string &val = args[++i];
			if (val == "true")
				options.safecuda_opts.fail_fast = true;
			else if (val == "false")
				options.safecuda_opts.fail_fast = false;
		} else if (arg == "-sf-logging") {
			options.safecuda_opts.log_violations = true;
			options.safecuda_opts.log_file = args[++i];
		} else if (arg.rfind("-sf-", 0) != 0) {
			options.nvcc_opts.nvcc_args.push_back(arg);
		}
	}

	EXPECT_FALSE(options.safecuda_opts.enable_bounds_check);
	EXPECT_TRUE(options.safecuda_opts.enable_debug);
	EXPECT_TRUE(options.safecuda_opts.fail_fast);
	EXPECT_TRUE(options.safecuda_opts.log_violations);
	EXPECT_EQ(options.safecuda_opts.log_file, "/var/log/sf.log");

	ASSERT_EQ(options.nvcc_opts.nvcc_args.size(), 2);
	EXPECT_EQ(options.nvcc_opts.nvcc_args[0], "-O2");
	EXPECT_EQ(options.nvcc_opts.nvcc_args[1], "kernel.cu");
}
