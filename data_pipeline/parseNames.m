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
            name_pieces = split(name, '-');
            if length(name_pieces) > 1
                name = strcat(name_pieces(1),"-" , name_pieces(2));
            end
        end
        newNames(n) = string(name);
    end
    names = newNames';
end