#pragma once

#include <utility>
#include <type_traits>

namespace simplex {

template<class Callable>
struct FilterFunction {
    Callable _filter;
    FilterFunction(Callable filter): _filter(std::move(filter)) {}
    
    template<class... Args>
    auto operator ()(Args&&... args) const noexcept {
        return _filter(std::forward<Args>(args)...);
    }
};

template<class Callable>
auto make_filter(Callable&& callable) noexcept {
    return FilterFunction<std::remove_reference_t<Callable>>(std::forward<Callable>(callable));
}

template<class A, class B>
auto operator || (const FilterFunction<A>& func_A, const FilterFunction<B>& func_B) noexcept {
    return make_filter([filter_A = func_A._filter, filter_B = func_B._filter](auto&&... args) -> auto {
        return filter_A(std::forward<decltype(args)>(args)...) || filter_B(std::forward<decltype(args)>(args)...);
    });
}

template<class A, class B>
auto operator || (FilterFunction<A>&& func_A, FilterFunction<B>&& func_B) noexcept {
    return make_filter([filter_A = std::move(func_A._filter), filter_B = std::move(func_B._filter)](auto&&... args) -> auto {
        return filter_A(std::forward<decltype(args)>(args)...) || filter_B(std::forward<decltype(args)>(args)...);
    });
}

template<class A, class B>
auto operator && (const FilterFunction<A>& func_A, const FilterFunction<B>& func_B) noexcept {
    return make_filter([filter_A = func_A._filter, filter_B = func_B._filter](auto&&... args) -> auto {
        return filter_A(std::forward<decltype(args)>(args)...) && filter_B(std::forward<decltype(args)>(args)...);
    });
}

template<class A, class B>
auto operator && (FilterFunction<A>&& func_A, FilterFunction<B>&& func_B) noexcept {
    return make_filter([filter_A = std::move(func_A._filter), filter_B = std::move(func_B._filter)](auto&&... args) -> auto {
        return filter_A(std::forward<decltype(args)>(args)...) && filter_B(std::forward<decltype(args)>(args)...);
    });
}

template<class A>
auto operator !(const FilterFunction<A>& func_A) noexcept {
    return make_filter([filter = func_A._filter](auto&&... args) -> auto {
        return !filter(std::forward<decltype(args)>(args)...);
    });
}

template<class A>
auto operator !(FilterFunction<A>&& func_A) noexcept {
    return make_filter([filter = std::move(func_A._filter)](auto&&... args) -> auto {
        return !filter(std::forward<decltype(args)>(args)...);
    });
}

template<class A, class B>
auto operator + (const FilterFunction<A>& func_A, const FilterFunction<B>& func_B) noexcept {
    return make_filter([filter_A = func_A._filter, filter_B = func_B._filter](auto&&... args) -> auto {
        return filter_A(std::forward<decltype(args)>(args)...) + filter_B(std::forward<decltype(args)>(args)...);
    });
}

template<class A, class B>
auto operator + (FilterFunction<A>&& func_A, FilterFunction<B>&& func_B) noexcept {
    return make_filter([filter_A = std::move(func_A._filter), filter_B = std::move(func_B._filter)](auto&&... args) -> auto {
        return filter_A(std::forward<decltype(args)>(args)...) + filter_B(std::forward<decltype(args)>(args)...);
    });
}

template<class A, class B>
auto operator * (const FilterFunction<A>& func_A, const FilterFunction<B>& func_B) noexcept {
    return make_filter([filter_A = func_A._filter, filter_B = func_B._filter](auto&&... args) -> auto {
        return filter_A(std::forward<decltype(args)>(args)...) * filter_B(std::forward<decltype(args)>(args)...);
    });
}

template<class A, class B>
auto operator * (FilterFunction<A>&& func_A, FilterFunction<B>&& func_B) noexcept {
    return make_filter([filter_A = std::move(func_A._filter), filter_B = std::move(func_B._filter)](auto&&... args) -> auto {
        return filter_A(std::forward<decltype(args)>(args)...) * filter_B(std::forward<decltype(args)>(args)...);
    });
}

template<class A>
auto operator -(const FilterFunction<A>& func_A) noexcept {
    return make_filter([filter = func_A._filter](auto&&... args) -> auto {
        return -filter(std::forward<decltype(args)>(args)...);
    });
}

template<class A>
auto operator -(FilterFunction<A>&& func_A) noexcept {
    return make_filter([filter = std::move(func_A._filter)](auto&&... args) -> auto {
        return -filter(std::forward<decltype(args)>(args)...);
    });
}

template<class A, class B>
auto operator >> (const FilterFunction<A>& func_A, const FilterFunction<B>& func_B) noexcept {
    return make_filter([filter_A = func_A._filter, filter_B = func_B._filter](auto&&... args) -> auto {
        return filter_B(filter_A(std::forward<decltype(args)>(args)...));
    });
}

template<class A, class B>
auto operator >> (FilterFunction<A>&& func_A, FilterFunction<B>&& func_B) noexcept {
    return make_filter([filter_A = std::move(func_A._filter), filter_B = std::move(func_B._filter)](auto&&... args) -> auto {
        return filter_B(filter_A(std::forward<decltype(args)>(args)...));
    });
}

}
