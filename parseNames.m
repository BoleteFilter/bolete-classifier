function names = parseNames(names)
    names = lower(names);
    for n = 1:length(names)
        entries = split(names(n), "(");
        name = entries(1);
        names(n) = name;
    end
    names = strip(names);
    names = replace(names, " ", "-");
    names = replace(names, "/", "");
    names = replace(names, ".", "");
    newNames = string.empty(length(names),0);
    for n = 1:length(names)
        name = names(n);
        if ~strcmp(name, "")
            name = split(name, ['-']);
            name = strcat(name(1),"-" , name(2));
        end
        newNames(n) = string(name);
    end
    names = newNames';
end