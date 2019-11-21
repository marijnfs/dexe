#pragma once

#include <ostream>
namespace dexe {

enum Code {
	FG_RED      = 31,
	FG_GREEN    = 32,
	FG_YELLOW   = 33,
	FG_BLUE     = 34,
	FG_PURPLE   = 35,
	FG_DEFAULT  = 39,
	BG_RED      = 41,
	BG_GREEN    = 42,
	BG_BLUE     = 44,
	BG_DEFAULT  = 49
};

class Colour {
	Code code;
 public:
 Colour(Code pCode) : code(pCode) {}
	friend std::ostream&
		operator<<(std::ostream& os, const Colour &mod) {
		return os << "\033[" << mod.code << "m";
	}
};

static Colour RED(FG_RED);
static Colour GREEN(FG_GREEN);
static Colour YELLOW(FG_YELLOW);
static Colour BLUE(FG_BLUE);
static Colour PURPLE(FG_PURPLE);
static Colour DEFAULT(FG_DEFAULT);

}