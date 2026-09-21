#ifndef YAML_HPP
#define YAML_HPP

#include <iostream>
#include <yaml-cpp/yaml.h>

namespace tools
{
    inline YAML::Node load(const std::string & path){
        try
        {
            return YAML::LoadFile(path);
        }catch(const YAML::BadFile & e)
        {
            std::cerr << e.what() << '\n';
            exit(1);
        }catch (const YAML::ParserException & e){
            std::cerr << e.what() << '\n';
            exit(1);
        }
        
    }
}; // namespace tools


#endif 