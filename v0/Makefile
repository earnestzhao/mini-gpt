
CC       := g++
CFLAGS   := -Wall -Wextra -O3 -std=c++11
INCLUDES := -I. -I..
BUILDDIR := build

CMDDIR  := cmd
LIBDIRS := common tokenizer embedding transformer
SRCDIRS := $(CMDDIR) $(LIBDIRS)
vpath %.cpp $(SRCDIRS)

LIBSRCS := $(wildcard $(addsuffix /*.cpp, $(LIBDIRS)))
LIBOBJS := $(foreach file, $(LIBSRCS), $(BUILDDIR)/$(subst /,_,$(file:.cpp=.o)))

CMDSRCS := $(wildcard $(CMDDIR)/*.cpp)
TARGETS := $(patsubst $(CMDDIR)/%.cpp, $(BUILDDIR)/%, $(CMDSRCS))

all: $(TARGETS)

$(BUILDDIR)/%: $(BUILDDIR)/$(CMDDIR)_%.o $(LIBOBJS)
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@

.SECONDEXPANSION:
$(BUILDDIR)/%.o: $$(shell echo $$* | sed 's/_/\//1').cpp
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf $(BUILDDIR)

.PHONY: clean
