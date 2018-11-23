#include <iostream>
#include <chrono>

#ifndef FSG_TRACE_H_
#define FSG_TRACE_H_

/** @file
 * Log system for C++ code, simple and convenient (for me at least).
 *
 * Features:
 *
 * - Easy to integrate into any project.
 * - From log, one-keypress jump to corresponding source.
 * - Supports various needs (from just "been there", easily log variable value,
 to arbitrary output).
 * - Each call case is as simple as you can rightfully imagine.
 *
 * Other characteristics:
 *
 * - No macro clash or anything thanks to specific prefix.
 * - Still just a plain text log, no *need* for special tools to parse.
 * - Each line informative yet short and easy to read, thanks to short pathname
 relative to compilation root (currently with
 * plain make and CMake projects).


    @code
    FSG_LOG_INIT__CALL_FROM_CPP_MAIN();

    FSG_LOG_MSG("Easy");

    int my_var = 42;
    FSG_LOG_VAR(my_var);

    FSG_LOG_MSG("Fill frobnicate " << my_var);

    FSG_LOG_MSG("See doc for other cases.");
    @endcode

 */

// Interface

int FSG_LOG_INDENTATION_LEVEL =
    0; /*HACK, won't work if 2 files include this file.*/

/** Before any log, for example at the start your main() function,
 * this outputs a conventional string (emacs convention) that allows
 * to jump to source code in a keypress. */
#ifdef FSG_PROJECT_ROOT
#define FSG_LOG_INIT__CALL_FROM_CPP_MAIN()                                     \
    std::cerr << "make: Entering directory '" << FSG_PROJECT_ROOT << "'"       \
              << std::endl
#else
#define FSG_LOG_INIT__CALL_FROM_CPP_MAIN()                                     \
    { /* When compiled with CMake or a Makefile, this will output to stderr:   \
         "make: Entering directory 'absolute compilation path'.   */           \
    }
#warning                                                                       \
    "FSG_PROJECT_RELATIVE_PATHNAME not defined, will not benefit from short paths in logs."
#endif

/** @name Items that Produce a full log line.
 */
///@{

/** Simplest "been there" message: just log "here" with file and line
 * info. */
#define FSG_LOG_LOCATION() FSG_LOG_MSG("here");

/** Convenience shortcut: simplest constant string log. */
#define FSG_LOG_MSG(TEXT)                                                      \
    FSG_LOG_BEGIN() << FSG_LOCATION() << FSG_INDENTATION() << TEXT             \
                    << FSG_LOG_END()

/** Log any variable (actually, any expression). */
#define FSG_LOG_VAR(VARNAME)                                                   \
    FSG_LOG_BEGIN() << FSG_LOCATION() << FSG_INDENTATION() << FSG_OSTREAM_VAR(VARNAME)              \
                    << FSG_LOG_END()

/** Convenience shortcut: for any class where operator<< is properly
 * defined, log a text and a dump of "this" object.  */
#define FSG_LOG_THIS(TEXT)                                                     \
    FSG_LOG_BEGIN() << FSG_LOCATION() << FSG_INDENTATION() << TEXT << ":" << this << FSG_LOG_END()

///@}

/** @name RAII-based scope tracing.

    For example:

    @code
    bar my_function(int foo)
    {
    FSG_TRACE_THIS_FUNCTION();
    int baz = foo + 2;
      {
        FSG_TRACE_THIS_SCOPE_WITH_LABEL("quux");
        int quux = baz + 3;
      }
    }
    @endcode
*/
///@{

#define FSG_TRACE_THIS_FUNCTION()                                              \
    Fidergo::Trace FSG_TRACE_this_function_tracer(                             \
        __PRETTY_FUNCTION__, FSG_CURRENT_FILE_NAME, __LINE__);
#define FSG_TRACE_THIS_SCOPE_WITH_STATIC_STRING(label)                         \
    Fidergo::Trace FSG_TRACE_this_scope_tracer(label, FSG_CURRENT_FILE_NAME,   \
                                               __LINE__);
#define FSG_TRACE_THIS_SCOPE_WITH_SSTREAM(expression)                          \
    auto ss = std::stringstream();                                             \
    ss << expression;                                                          \
    Fidergo::Trace FSG_TRACE_this_scope_tracer(                                \
        ss.str(), FSG_CURRENT_FILE_NAME, __LINE__);

///@}

/** @name Helpers that can be used as part of a custom log line.

    For example:

    @code
    FSG_LOG_BEGIN() << FSG_LOCATION() << FSG_INDENTATION() << "My text"
    << FSG_OSTREAM_VAR(my_variable ) << " and "
    << FSG_OSTREAM_VAR( my_other_variable ) << FSG_LOG_END();
    @endcode

 */
///@{

#define FSG_LOG_BEGIN() std::cerr

#define FSG_LOG_END() std::endl

#define FSG_LONGEMPTY_STRING                                                   \
    "                                                                "

#define FSG_LONGEMPTY_STRING_SIZE (sizeof(FSG_LONGEMPTY_STRING))

#define FSG_LOCATION() FSG_CURRENT_FILE_NAME << ":" << __LINE__ << ":"

#define FSG_INDENTATION()                                                      \
    (FSG_LONGEMPTY_STRING +                                                    \
     (FSG_LONGEMPTY_STRING_SIZE - 4 * FSG_LOG_INDENTATION_LEVEL - 1))

#define FSG_OSTREAM_VAR(VARNAME) #VARNAME << " = " << (VARNAME)

#define FSG_OSTREAM_FIELD(OBJECT, FIELDNAME)                                   \
    #FIELDNAME << "=" << OBJECT.FIELDNAME << " "

#define FSG_OSTREAM_POINTED_FIELD(OBJECT, FIELDNAME)                           \
    #FIELDNAME << "=" << OBJECT->FIELDNAME << " "

///@}

// CMAKE_CURRENT_LIST_DIR
// CMAKE_CURRENT_SOURCE_DIR
// CMAKE_HOME_DIRECTORY
// CMAKE_SOURCE_DIR
// PROJECT_SOURCE_DIR
// myprojectname_SOURCE_DIR

// Internal implementation details

#ifdef FSG_PROJECT_RELATIVE_PATHNAME
#define FSG_CURRENT_FILE_NAME FSG_PROJECT_RELATIVE_PATHNAME
#else
#define FSG_CURRENT_FILE_NAME __FILE__
#warning                                                                       \
    "FSG_PROJECT_RELATIVE_PATHNAME not defined, will not benefit from short paths in logs."
#endif

namespace Fidergo {

class Trace {
  private:
    std::string scopeName;
    const char *file_name;
    int line;

    // https://en.cppreference.com/w/cpp/chrono/time_point
    std::chrono::time_point<std::chrono::steady_clock> time_start;

  public:
    Trace(const std::string &ScopeName, const char *file_name, int line)
        : file_name(file_name), line(line) {
        this->scopeName = ScopeName;

        FSG_LOG_BEGIN() << std::endl
                        << file_name << ":" << line << ":" << FSG_INDENTATION()
                        << "{ // Entered: " << scopeName << FSG_LOG_END();

        ++FSG_LOG_INDENTATION_LEVEL;

        time_start = std::chrono::steady_clock::now();
    }

    ~Trace() {
        std::chrono::time_point<std::chrono::steady_clock> time_end = std::chrono::steady_clock::now();

        auto time_interval = time_end - time_start;
        auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(time_interval).count();
    
        --FSG_LOG_INDENTATION_LEVEL;

        FSG_LOG_BEGIN() << file_name << "-" << line << "-" << FSG_INDENTATION()
                        << "} //  Exited: " << scopeName << " -- Duration/µs: " << microseconds << std::endl << FSG_LOG_END();
    }
};
}

#endif // FSG_TRACE_H_
