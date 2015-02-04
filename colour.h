#ifndef __COLOUR_H__
#define __COLOUR_H__

#include <ostream>
enum Code {
	FG_RED      = 31,
	FG_GREEN    = 32,
	FG_BLUE     = 34,
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

Colour RED(FG_RED);
Colour GREEN(FG_GREEN);
Colour BLUE(FG_BLUE);
Colour DEFAULT(FG_DEFAULT);

#endif
